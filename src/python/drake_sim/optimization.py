from typing import Tuple

import numpy as np
from pydrake.all import (
    LeafSystem, BasicVector, MathematicalProgram, Solve,
    JacobianWrtVariable, AbstractValue,
    MultibodyPlant, ModelInstanceIndex,
    RigidTransform, SpatialVelocity,
)


class OptimizeTrajectory(LeafSystem):
    """
    入力ポート
    - "iiwa_state" : BasicVector[2*nq]
        現在の関節状態ベクトル [q; v] （q: 位置, v: 速度）。
    - "goal_transform" : Abstract (RigidTransform)
        ストライク時のエンドエフェクタ目標姿勢（パドル位置・姿勢）。
    - "goal_spatial_velocity" : Abstract (SpatialVelocity)
        ストライク時のエンドエフェクタ目標空間速度（特に並進速度）。
    - "strike_time" : BasicVector[1]
        現在からストライクまでの時間 ts [s]。
        （論文の t_s。ホライゾン T 内にあると仮定）

    出力ポート
    - "desired_state" : BasicVector[2*nq]
        現在ステップでの理想状態 [q_des; v_des]。
        JointStiffnessOptimizationController が追従すべき値。
    - "qdd_desired" : BasicVector[nq]
        現在ステップでの feedforward 関節加速度 qdd_des。
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        iiwa_instance: ModelInstanceIndex,
        end_effector_frame_name: str,
        normal_offset_in_ee: np.ndarray,
        horizon: float = 0.6,
        N_ctrl: int = 8,
        num_limit_samples: int = 10,
        W_a: float = 1.0,
        W_r: float = 1e-2,
        W_n: float = 1.0,
        max_sqp_iters: int = 5,
    ):
        """
        Parameters
        plant :
            既に Finalize 済みの MultibodyPlant。
        iiwa_instance :
            iiwa モデルの ModelInstanceIndex。
        end_effector_frame_name :
            パドルが付いているエンドエフェクタフレームの名前。標準の名前を忘れたため。
        normal_offset_in_ee :
            パドル面上の「法線方向の点」へのオフセット [x,y,z]（エンドエフェクタ座標）。
            論文の K_n(q) を作るために、K_p(q) と K_n(q) の差を法線ベクトルとみなす。
        horizon :
            プランニングホライゾン T [s]。
        N_ctrl :
            Bezier control point の個数（論文では 8）。
        num_limit_samples :
            関節リミットを課す時間サンプル数（(0, T] を等間隔にサンプリング）。
        W_a, W_r :
            コスト f_a, f_r の重み。
        W_n :
            法線方向の誤差 ||n(q_s) - n_des||^2 の重み。
        max_sqp_iters :
            1 ステップで回す SQP 反復回数。
        """
        super().__init__()
        self._plant = plant
        self._iiwa = iiwa_instance
        self._ee_frame = plant.GetFrameByName(end_effector_frame_name, iiwa_instance)
        self._normal_offset = np.asarray(normal_offset_in_ee, dtype=float)
        assert self._normal_offset.shape == (3,)

        self._horizon = horizon
        self._N = N_ctrl
        self._order = N_ctrl - 1
        self._num_limit_samples = num_limit_samples
        self._W_a = W_a
        self._W_r = W_r
        self._W_n = W_n
        self._max_sqp_iters = max_sqp_iters

        # 関節数とリミット取得
        self._nq = plant.num_positions(iiwa_instance)
        self._nv = plant.num_velocities(iiwa_instance)
        assert self._nq == self._nv  # iiwa はそうなっているはず

        # self._q_min = np.array([
        #     -2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543,
        # ])
        # self._q_max = np.array([
        #     2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944,  3.0543,
        # ])
        # 適当、よくわからないので、後の制約式まで含めて一旦関節制約はコメントアウトした


        # rest 姿勢（デフォルトポーズ）
        self._q_rest = plant.GetDefaultPositions(iiwa_instance)

        #入出力ポート宣言

        # 1. iiwa_state: [q; v]
        self.DeclareVectorInputPort("iiwa_state", BasicVector(2 * self._nq))

        # 2. goal_transform: RigidTransform
        self.DeclareAbstractInputPort(
            "goal_transform",
            AbstractValue.Make(RigidTransform())
        )

        # 3. goal_spatial_velocity: SpatialVelocity
        self.DeclareAbstractInputPort(
            "goal_spatial_velocity",
            AbstractValue.Make(SpatialVelocity())
        )

        # 4. strike_time: scalar ts
        self.DeclareVectorInputPort("strike_time", BasicVector(1))

        # 5. desired_state: [q_des; v_des]
        self.DeclareVectorOutputPort(
            "desired_state",
            BasicVector(2 * self._nq),
            self.CalcDesiredState,
        )

        # 6. qdd_desired
        self.DeclareVectorOutputPort(
            "qdd_desired",
            BasicVector(self._nq),
            self.CalcDesiredAcceleration,
        )

    # Bezier 基底関数（論文中の B(·, t), B'(·, t) に対応）

    def _bezier_basis(self, tau: float) -> np.ndarray:
        """
        7 次 Bezier（control point N_ctrl 個）の基底ベクトル b(tau) を返す。

        Parameters
        tau : float
            正規化時間 in [0, 1]  （tau = t / T）。

        Returns
        b : np.ndarray, shape = (N_ctrl,)
            b_k(tau) = C(order, k) * tau^k * (1 - tau)^{order-k}
        """
        from math import comb
        n = self._order
        return np.array(
            [comb(n, k) * tau**k * (1 - tau) ** (n - k) for k in range(n + 1)]
        )

    def _bezier_basis_derivative(self, tau: float) -> np.ndarray:
        """
        Bezier 基底の tau 微分 db/dtau を返す。
        B'(q_c, tau) = q_c @ db/dtau で qdot を得る。

        ※実時間微分は (d/dt) = (1/T) * (d/dtau) になるが、
        論文の実装と同様、比例定数はコストの重みで吸収できる前提で
        ここでは tau 微分のまま使う。
        """
        from math import comb
        n = self._order
        db = []
        for k in range(n + 1):
            c = comb(n, k)
            # d/dtau [tau^k (1−tau)^{n-k}]
            if k == 0:
                term1 = 0.0
            else:
                term1 = k * tau ** (k - 1) * (1 - tau) ** (n - k)
            if k == n:
                term2 = 0.0
            else:
                term2 = -(n - k) * tau**k * (1 - tau) ** (n - k - 1)
            db.append(c * (term1 + term2))
        return np.array(db)

    # 1ステップ分の SQP：非線形制約を線形化して QP を解く

    def _solve_sqp_once(
        self,
        q0: np.ndarray,
        v0: np.ndarray,
        ts: float,
        goal_T: RigidTransform,
        goal_V: SpatialVelocity,
        q_lin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        与えられた線形化点 q_lin の周りで1回だけ QP を解き、
        Bezier control point と strike 状態を更新する。

        Parameters
        q0, v0 :
            現在の関節位置・速度。
        ts :
            ストライク時刻 [s]。
        goal_T, goal_V :
            ストライク時の EE 目標 Pose, SpatialVelocity。
        q_lin :
            EE 関連制約を線形化するための joint 配列（通常は前回の q_s）。

        Returns
        q_ctrl : np.ndarray, shape = (nq, N_ctrl)
            最適化された Bezier control points。
        q_s : np.ndarray, shape = (nq,)
            ストライク時の関節位置。
        qsdot : np.ndarray, shape = (nq,)
            ストライク時の関節速度。
        qdd0 : np.ndarray, shape = (nq,)
            control point の 2 階差分から得た「初期付近の加速度 proxy」。
            → qdd_des として使う。
        """
        nq, N = self._nq, self._N
        T = self._horizon
        tau_s = np.clip(ts / T, 0.0, 1.0)



        prog = MathematicalProgram()

        # 決定変数
        q_ctrl = prog.NewContinuousVariables(nq, N, "q_ctrl")     # Bezier control points
        q_s = prog.NewContinuousVariables(nq, 1, "q_s").reshape((nq,))
        qsdot = prog.NewContinuousVariables(nq, 1, "qsdot").reshape((nq,))

        # Bezier を評価するヘルパー
        def bezier(qc, basis):
            # qc: (nq, N), basis: (N,) → (nq,)
            return qc @ basis

        B0 = self._bezier_basis(0.0)
        dB0 = self._bezier_basis_derivative(0.0)
        Bs = self._bezier_basis(tau_s)
        if np.any(np.isnan(Bs)):
            raise ValueError("NaN in Bezier basis at tau_s")
        dBs = self._bezier_basis_derivative(tau_s)
        print(f"tau_s={tau_s}, ts={ts}, T={T}")

        # 初期条件 (4b),(4c)
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, B0) - q0, np.zeros(nq))
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, dB0) - v0, np.zeros(nq))

        # slack とのリンク (5a),(5b)
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, Bs) - q_s, np.zeros(nq))
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, dBs) - qsdot, np.zeros(nq))

        # joint limit (4d): 時間サンプリング
        # if self._num_limit_samples > 0:
        #     taus = np.linspace(0.0, 1.0, self._num_limit_samples + 1)[1:]  # 0 は除外
        #     for tau in taus:
        #         Bt = self._bezier_basis(tau)
        #         qt = bezier(q_ctrl, Bt)
        #         for i in range(nq):
        #             prog.AddLinearConstraint(qt[i] >= self._q_min[i])
        #             prog.AddLinearConstraint(qt[i] <= self._q_max[i])

        # EE 関連制約 (4f),(4g),(4h) を q_lin で線形化
        context = self._plant.CreateDefaultContext()
        if len(q_lin) != self._plant.num_positions(self._iiwa):
            print(f"len(q_lin)={len(q_lin)}, expected={self._plant.num_positions(self._iiwa)}")
            q_lin = np.concatenate(
                [q_lin, np.zeros(self._plant.num_positions(self._iiwa) - len(q_lin))]
            )
        self._plant.SetPositions(context, self._iiwa, q_lin)

        # 目標位置・速度・法線
        p_des = goal_T.translation()                  # R^3
        v_des = goal_V.translational()                # R^3

        # 中心点 K_p(q)
        p0 = self._plant.CalcPointsPositions(
            context, self._ee_frame, [0.0, 0.0, 0.0], self._plant.world_frame()
        ).ravel()
        Jp = self._plant.CalcJacobianTranslationalVelocity(
            context, JacobianWrtVariable.kQDot,
            self._ee_frame, [0.0, 0.0, 0.0],
            self._plant.world_frame(), self._plant.world_frame()
        )[:, :nq]

        # 「法線方向の点」 K_n(q) = EE フレームの normal_offset_in_ee をワールドに写した点
        pn0 = self._plant.CalcPointsPositions(
            context, self._ee_frame, self._normal_offset, self._plant.world_frame()
        ).ravel()
        Jpn = self._plant.CalcJacobianTranslationalVelocity(
            context, JacobianWrtVariable.kQDot,
            self._ee_frame, self._normal_offset,
            self._plant.world_frame(), self._plant.world_frame()
        )[:, :nq]

        # n(q) = K_n(q) - K_p(q) の線形近似
        n0 = pn0 - p0                    # 法線ベクトル（正規化はしない）
        Jn = Jpn - Jp                    # dn/dq

        # 位置: K_p(q_s) ≈ p0 + Jp (q_s - q_lin) = p_des
        p_approx = p0 + Jp @ (q_s - q_lin)
        prog.AddLinearEqualityConstraint(p_approx - p_des, np.zeros(3))

        # 速度: J_p(q_s) qsdot ≈ Jp(q_lin) qsdot = v_des
        v_approx = Jp @ qsdot
        prog.AddLinearEqualityConstraint(v_approx - v_des, np.zeros(3))

        # 法線: n(q_s) ≈ n0 + Jn (q_s - q_lin) が n_des に近いように
        n_des = goal_T.rotation().matrix() @ np.array([0.0, 0.0, 1.0])
        n_approx = n0 + Jn @ (q_s - q_lin)
        # → QP のコストに ||n_approx - n_des||^2 を入れる（4h の soft 版）

        # コスト: f_r + f_a + 法線誤差
        cost = 0.0

        # rest 姿勢からのずれ f_r (式 (7) 相当)
        for i in range(nq):
            for k in range(N):
                diff = q_ctrl[i, k] - self._q_rest[i]
                cost += self._W_r * diff * diff

        # control point の 2 階差分による滑らかさ f_a (式 (6) 相当)
        for i in range(nq):
            for k in range(1, N - 1):
                dd = q_ctrl[i, k + 1] - 2 * q_ctrl[i, k] + q_ctrl[i, k - 1]
                cost += self._W_a * dd * dd

        # 法線の soft 制約 ||n(q_s) - n_des||^2
        for j in range(3):
            diff_n = n_approx[j] - n_des[j]
            cost += self._W_n * diff_n * diff_n

        prog.AddQuadraticCost(cost)

        # QP を解く
        result = Solve(prog)
        if not result.is_success():
            # 失敗したら単純に「その場にとどまる」ような出力にする
            q_ctrl_zero = np.tile(q0.reshape(-1, 1), (1, N))
            qdd0_zero = np.zeros(nq)
            return q_ctrl_zero, q0, v0, qdd0_zero

        q_ctrl_sol = result.GetSolution(q_ctrl)
        q_s_sol = result.GetSolution(q_s)
        qsdot_sol = result.GetSolution(qsdot)

        # 「初期付近の加速度 proxy」: 先頭 3 点の 2 階差分
        qdd0 = np.zeros(nq)
        for i in range(nq):
            qdd0[i] = q_ctrl_sol[i, 2] - 2 * q_ctrl_sol[i, 1] + q_ctrl_sol[i, 0]

        return q_ctrl_sol, q_s_sol, qsdot_sol, qdd0

    # 1ステップ分の MPC（SQP ループ含む）

    def _solve_mpc(
        self,
        q0: np.ndarray,
        v0: np.ndarray,
        ts: float,
        goal_T: RigidTransform,
        goal_V: SpatialVelocity,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SQP を max_sqp_iters 回まわし、論文スタイルの Kinematic MPC を解く。

        戻り値は (q_des, v_des, qdd_des)。
        今のステップでは、Bezier 軌道を t=0（またはごく小さい時刻）で評価
        した値を desired として出す。
        """
        # 線形化点の初期値：現在の joint
        q_lin = q0.copy()
        q_ctrl = None
        qdd0 = np.zeros_like(q0)

        for _ in range(self._max_sqp_iters):
            try:
                q_ctrl, q_s, qsdot, qdd0 = self._solve_sqp_once(
                    q0, v0, ts, goal_T, goal_V, q_lin
                )
            except ValueError:
                # NaN が出たらループを抜ける
                return q0, v0, qdd0
            # 次の線形化点として q_s を使う（論文と同じ発想）
            q_lin = q_s

        # 最終的な Bezier から「現在時刻の desired」を取り出す
        # t = 0 では初期条件拘束で q_des=q0 なので、
        # 少しだけ先の時刻で評価する
        T = self._horizon
        t_now = min(0.01, T)   # 1e-2[s] だけ先
        tau_now = t_now / T
        B_now = self._bezier_basis(tau_now)
        dB_now = self._bezier_basis_derivative(tau_now)

        q_des = q_ctrl @ B_now
        v_des = q_ctrl @ dB_now

        # 加速度は qdd0 をそのまま使う（軌道先頭付近の加速度）
        qdd_des = qdd0

        return q_des, v_des, qdd_des

    # 出力ポート計算

    def CalcDesiredState(self, context, output):
        """
        出力ポート "desired_state" を計算する。
        入力ポートから現在状態・ストライク目標・ts を取得し、
        Kinematic MPC を解いて [q_des; v_des] を出力する。
        """
        state = self.get_input_port(0).Eval(context)
        q0 = np.asarray(state[: self._nq])
        v0 = np.asarray(state[self._nq : 2 * self._nq])

        goal_T = self.get_input_port(1).Eval(context)
        goal_V = self.get_input_port(2).Eval(context)
        ts = float(self.get_input_port(3).Eval(context)[0])

        q_des, v_des, _ = self._solve_mpc(q0, v0, ts, goal_T, goal_V)

        y = np.zeros(2 * self._nq)
        y[: self._nq] = q_des
        y[self._nq :] = v_des
        #print(f"q_des={q_des}, v_des={v_des}")
        output.SetFromVector(y)

    def CalcDesiredAcceleration(self, context, output):
        """
        出力ポート "qdd_desired" を計算する。
        （現状は CalcDesiredState と同じ MPC をもう一度解いている。
          実用上はキャッシュする方が望ましいが、
          理論構造は論文どおりなので、まずはこの形で。）
        """
        state = self.get_input_port(0).Eval(context)
        q0 = np.asarray(state[: self._nq])
        v0 = np.asarray(state[self._nq : 2 * self._nq])

        goal_T = self.get_input_port(1).Eval(context)
        goal_V = self.get_input_port(2).Eval(context)
        ts = float(self.get_input_port(3).Eval(context)[0])

        _, _, qdd_des = self._solve_mpc(q0, v0, ts, goal_T, goal_V)
        output.SetFromVector(qdd_des)
