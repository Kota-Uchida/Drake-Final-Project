from typing import Tuple

import numpy as np
from pydrake.all import (
    AbstractValue,
    BasicVector,
    JacobianWrtVariable,
    LeafSystem,
    MathematicalProgram,
    ModelInstanceIndex,
    MultibodyForces,
    MultibodyPlant,
    RigidTransform,
    Solve,
    SpatialVelocity,
    eq,
    ge,
    le,
    Cylinder,
    Sphere,
    Rgba,
    RotationMatrix,
    BsplineBasis,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    PositionConstraint,
    RigidTransform,
    OrientationConstraint

)
from pydrake.systems.framework import Context


class SimpleController(LeafSystem):
    """
    A simple PD controller for a 7-DOF robotic arm.
    """
    def __init__(self, plant: MultibodyPlant,k_p: float, k_i: float, k_d: float):
        super().__init__()
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa_idx = plant.GetModelInstanceByName("iiwa")
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.DeclareVectorInputPort(name="iiwa_state", size=14)
        self.input_port = self.get_input_port(0)
        self.q_desired = self.DeclareVectorInputPort(name="desired_state", size=14)
        self.DeclareVectorOutputPort(name="iiwa_torque", size=7, calc=self.ComputeTorque)
        self.output_port = self.get_output_port(0)
        self.previous_q = None
        self.integral_error = np.zeros(7)
        self.previous_time = 0.0
        self.count = 0

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        q = self.input_port.Eval(context)[:7]
        q_d = self.input_port.Eval(context)[7:14]
        q_desired = self.q_desired.Eval(context)[:7]
        qd_desired = self.q_desired.Eval(context)[7:14]
        current_time = context.get_time()
        dt = current_time - self.previous_time
        # M = self.plant.CalcMassMatrix(plant_context)
        # M = M[:7, :7]
        # C = self.plant.CalcBiasTerm(plant_context)
        # C = C[:7]
        self.plant.SetPositionsAndVelocities(self.plant_context, self.iiwa_idx, np.hstack((q, q_d)))
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        tau_g = tau_g[:7]

        if dt > 0:
            self.integral_error += (q_desired - q) * dt

        torque = self.k_p * (q_desired - q) + self.k_i * self.integral_error + self.k_d * (qd_desired - q_d) - tau_g
        self.previous_q = q.copy()
        self.previous_time = current_time
        output.SetFromVector(torque)

        # if self.count % 100 == 0:
        #     print(f"Time: {current_time:.2f}, q: {q}, q_desired: {q_desired}, torque: {torque}, tau_g: {tau_g}, k_p*(q_desired - q): {self.k_p * (q_desired - q)}, k_d*(qd_desired - q_d): {self.k_d * (qd_desired - q_d)}")
        # self.count += 1
        
    # def ComputeTorque(self, context: Context, output: BasicVector) -> None:
    #     # Get current state
    #     state = self.get_input_port(0).Eval(context)
    #     q = state[:7]
    #     q_dot = state[7:14]

    #     # Get desired state
    #     desired = self.get_input_port(1).Eval(context)
    #     q_desired = desired[:7]
    #     qd_desired = desired[7:14]

    #     # Time step for integral
    #     current_time = context.get_time()
    #     dt = current_time - self.previous_time if self.previous_time > 0 else 0.0

    #     # Update integral error
    #     if dt > 0:
    #         self.integral_error += (q_desired - q) * dt

    #     # PD control + gravity compensation
    #     tau = (
    #         self.k_p * (q_desired - q) +
    #         self.k_d * (qd_desired - q_dot) +
    #         self.k_i * self.integral_error +
    #         self.plant.CalcGravityGeneralizedForces(context)[:7]  # gravity only
    #     )

    #     # Update previous time
    #     self.previous_time = current_time

    #     # Output torque
    #     output.SetFromVector(tau)

class WSGController(LeafSystem):
    """
    A simple PD controller for a Weiss WSG gripper.
    """
    def __init__(self, position_desired: np.ndarray, k_p: float=8.0, k_i: float=20.0, k_d: float=100.0, max_torque: float = 1000.0):
        super().__init__()
        self.position_desired = np.asarray(position_desired, dtype=float)
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.max_torque = float(max_torque)

        self.DeclareVectorInputPort(name="wsg_state", size=4)
        self.input_port = self.get_input_port(0)
        self.DeclareVectorOutputPort(name="wsg_torque", size=2, calc=self.ComputeTorque)
        self.output_port = self.get_output_port(0)

        self.previous_position = None
        self.integral_error = np.zeros(2, dtype=float)
        self.previous_time = None
        # Optional integrator limits to avoid windup
        self.integrator_limit = 1.0 * np.ones(2, dtype=float)

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        # Read positions robustly and take first two values (finger positions)
        raw = self.input_port.Eval(context)
        positions = np.asarray(raw, dtype=float)[:2]

        current_time = float(context.get_time())
        if self.previous_time is None:
            dt = 0.0
        else:
            dt = current_time - self.previous_time

        # Compute velocities only if time advanced; avoid divide-by-zero
        if dt > 0.0:
            if self.previous_position is not None:
                velocities = (positions - self.previous_position) / dt
            else:
                velocities = np.zeros(2, dtype=float)

            # Integrate with anti-windup (clamp integral)
            self.integral_error += (self.position_desired - positions) * dt
            self.integral_error = np.clip(self.integral_error, -self.integrator_limit, self.integrator_limit)
        else:
            velocities = np.zeros(2, dtype=float)

        # PD + I control
        torque = self.k_p * (self.position_desired - positions) + self.k_i * self.integral_error - self.k_d * velocities

        # Sanitize NaN/Inf and clamp to safe range before sending to plant
        torque = np.nan_to_num(torque, nan=0.0, posinf=self.max_torque, neginf=-self.max_torque)
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # Update stored state for next step
        self.previous_position = positions.copy()
        self.previous_time = current_time

        output.SetFromVector(torque)

# class OptimizeTrajectory(LeafSystem):
#     """
#     Calculates optimal trajectory in the parameter space of a 7-DOF robotic arm.
#     Trajectory is optimized in the internal function and the current desired joint configurations and velocities are outputted.
#     inputs:
#         - goal_transform(RigidTransform): RigidTransform representing desired end-effector pose
#         - goal_spatial_velocity(SpatialVelocity): SpatialVelocity representing desired end-effector velocity
#     outputs:
#         - q_desired (7x1 Vector): Desired joint positions
#         - qd_desired (7x1 Vector): Desired joint velocities
#     """
#     def __init__(self, plant: MultibodyPlant):
#         super().__init__()
#         self.plant = plant
#         self.plant_context = plant.CreateDefaultContext()

#         # Inputs
#         self.DeclareAbstractInputPort(
#             name="goal_transform",
#             model_value=AbstractValue.Make(RigidTransform())
#         )
#         self.DeclareAbstractInputPort(
#             name="goal_spatial_velocity",
#             model_value=AbstractValue.Make(SpatialVelocity())
#         )

#         # Outputs
#         self.DeclareVectorOutputPort(
#             name="desired_state",
#             size=14,
#             calc=self.ComputeDesiredState
#         )
#         self.DeclareVectorOutputPort(
#             name="qdd_desired",
#             size=7,
#             calc=self.CalcQddDesired
#         )

#         # Internal storage
#         self._q_desired = np.zeros(7)
#         self._qd_desired = np.zeros(7)
#         self._qdd_desired = np.zeros(7)

#     def solve_trajectory_optimization(self, context: Context):
#         """
#         Solve a simple quadratic program to find joint positions and velocities that
#         achieve the desired end-effector pose and spatial velocity.
#         """
#         goal_X = self.get_input_port(0).Eval(context)
#         goal_V = self.get_input_port(1).Eval(context)

#         prog = MathematicalProgram()
#         q = prog.NewContinuousVariables(7, "q")
#         qd = prog.NewContinuousVariables(7, "qd")

#         # Update context with candidate q, qd (symbolic)
#         print("num_positions:", self.plant.num_positions())
#         print("num_velocities:", self.plant.num_velocities())
#         self.plant.SetPositions(self.plant_context, np.zeros(16))  # initial guess
#         self.plant.SetVelocities(self.plant_context, np.zeros(15))

#         # End-effector frame
#         ee_frame = self.plant.GetBodyByName("iiwa_link_7").body_frame()
#         world_frame = self.plant.world_frame()

#         # Jacobian at nominal position (linearization)
#         J = self.plant.CalcJacobianSpatialVelocity(
#             self.plant_context,
#             JacobianWrtVariable.kV,
#             ee_frame,
#             np.zeros(3),
#             world_frame,
#             world_frame
#         )[:6, :7]  # 6x7 Jacobian

#         # Linear cost: stay near current pose
#         prog.AddQuadraticCost(q @ q)  # minimize magnitude of q
#         prog.AddQuadraticCost(qd @ qd)  # minimize velocities

#         # Constraints: end-effector position & velocity
#         # Here, linearize position via Jacobian (6x7) to approximate pose error
#         ee_pos = goal_X.translation()  # 3D target
#         current_pos = self.plant.CalcRelativeTransform(
#             self.plant_context, frame_A=world_frame, frame_B=ee_frame
#         ).translation()
#         pos_error = ee_pos - current_pos
#         prog.AddLinearConstraint(J[:3, :] @ q == pos_error)  # position
#         prog.AddLinearConstraint(J[3:, :] @ qd == goal_V.rotational())  # angular velocity

#         # Solve
#         result = Solve(prog)
#         if not result.is_success():
#             raise RuntimeError("Trajectory optimization failed.")

#         self._q_desired = result.GetSolution(q)
#         self._qd_desired = result.GetSolution(qd)

#         # For simplicity, desired acceleration is PD towards goal
#         k_p = 100.0
#         k_d = 20.0
#         self._qdd_desired = k_p * (self._q_desired - np.zeros(7)) + k_d * (self._qd_desired - np.zeros(7))

#     def ComputeDesiredState(self, context: Context, output: BasicVector) -> None:
#         self.solve_trajectory_optimization(context)
#         desired_state = np.concatenate([self._q_desired, self._qd_desired])
#         output.SetFromVector(desired_state)

#     def CalcQddDesired(self, context: Context, output: BasicVector) -> None:
#         output.SetFromVector(self._qdd_desired)


    
class JointStiffnessOptimizationController(LeafSystem):
    """"
    Joint stiffness controller that computes torques using optimization to achieve desired joint positions and velocities.
    Constructed with a MultibodyPlant representing the robotic arm.
    inputs:
        - estimated_state (14x1 Vector): Current joint positions and velocities
        - desired_state (14x1 Vector): Desired joint positions and velocities
    outputs:
        - torque (7x1 Vector): Computed joint torques

    optimization:
        minimize(q_dd, tau) ||q_dd-q_dd_desired||^2
        subject to:
        M(q) * q_dd + C(q, qd) = tau_g + f_ext + B * tau 
    """
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.iiwa_idx = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()
        self._B = plant.MakeActuationMatrix()[:7, :7]
        self.f_ext = MultibodyForces(plant)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()
        self.q_dd = np.zeros(7)
        self._initial_guess = np.zeros(14)

        nq = 7
        self.DeclareVectorInputPort(name="estimated_state", size=2*nq)
        self.DeclareVectorInputPort(name="desired_state", size=2*nq)
        self.DeclareVectorOutputPort(name="torque", size=nq, calc=self.ComputeTorque)

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        current_state = self.get_input_port(0).Eval(context)
        desired_state = self.get_input_port(1).Eval(context)
        n_dof = 7
        q = current_state[:n_dof]
        qd = current_state[n_dof:2*n_dof]
        q_desired = desired_state[:n_dof]
        qd_desired = desired_state[n_dof:2*n_dof]

        # Update plant context
        self.plant.SetPositions(self.plant_context, self.iiwa_idx, q)
        self.plant.SetVelocities(self.plant_context, self.iiwa_idx, qd)

        # Mass, bias, gravity, external forces
        M_matrix = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
        M_matrix = M_matrix[self.iiwa_start:self.iiwa_end+1, self.iiwa_start:self.iiwa_end+1]

        C_matrix = self.plant.CalcBiasTerm(self.plant_context)
        C_matrix = C_matrix[self.iiwa_start:self.iiwa_end+1]

        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        tau_g = tau_g[self.iiwa_start:self.iiwa_end+1]

        self.plant.CalcForceElementsContribution(self.plant_context, self.f_ext)
        f_ext = self.f_ext.generalized_forces()[self.iiwa_start:self.iiwa_end+1]

        # Optimization
        prog = MathematicalProgram()
        tau = prog.NewContinuousVariables(n_dof, "tau")
        q_dd = prog.NewContinuousVariables(n_dof, "q_dd")

        # Desired acceleration (PD term)
        k_p = 100.0
        k_d = 20.0
        q_dd_desired = k_p * (q_desired - q) + k_d * (qd_desired - qd)

        # Quadratic cost
        prog.AddQuadraticCost((q_dd - q_dd_desired) @ (q_dd - q_dd_desired))

        # Linear equality: M q_dd - B tau = tau_g + f_ext - C
        A_dyn = np.hstack([M_matrix, -self._B])
        b_dyn = (tau_g + f_ext - C_matrix).reshape(-1, 1)
        prog.AddLinearEqualityConstraint(A_dyn, b_dyn, np.concatenate([q_dd, tau]))

        # Solve
        result = Solve(prog, initial_guess=self._initial_guess)
        if not result.is_success():
            raise RuntimeError("Optimization failed to find a solution.")

        optimal_tau = result.GetSolution(tau)
        output.SetFromVector(optimal_tau)
        self._initial_guess = np.concatenate([result.GetSolution(q_dd), optimal_tau])


# class SimpleOptimizeTrajectory(LeafSystem):
#     """
#     A simple trajectory optimizer that outputs desired joint positions and velovities.
#     Use default KinematicTrajectoryOptimization from Drake.
#     input ports:
#         - "iiwa_state" : BasicVector[2*nq]
#             現在の関節状態ベクトル [q; v] （q: 位置, v: 速度）。
#         - "goal_transform" : Abstract (RigidTransform)
#             ストライク時のエンドエフェクタ目標姿勢（パドル位置・姿勢）。
#         - "goal_spatial_velocity" : Abstract (SpatialVelocity)
#             ストライク時のエンドエフェクタ目標空間速度（特に並進速度）。
#         - "strike_time" : BasicVector[1]
#             現在からストライクまでの時間 ts [s]。
#             （論文の t_s。ホライゾン T 内にあると仮定）
#     output ports:

#     """
#     def __init__(
#             self,
#             plant: MultibodyPlant,
#             iiwa_instance: ModelInstanceIndex,
#             controller_plant: MultibodyPlant,
#             end_effector_frame_name: str,
#             normal_offset_in_ee: np.ndarray,
#             horizon: float = 1.5,
#             N_ctrl: int = 20,
#             num_limit_samples: int = 10,
#             max_sqp_iters: int = 5,
#             meshcat = None,
#             logger = None,
#     ):
#         super().__init__()
#         self._plant = plant
#         self._iiwa = iiwa_instance
#         self._controller_plant = controller_plant
#         self.ee_frame = controller_plant.GetFrameByName(end_effector_frame_name, iiwa_instance)
#         self._normal_offset = np.asarray(normal_offset_in_ee, dtype=float)
#         assert self._normal_offset.shape == (3,)

#         self.horizon = horizon
#         self.N = N_ctrl
#         self.num_limit_samples = num_limit_samples
#         self._max_sqp_iters = max_sqp_iters
#         self._meshcat = meshcat
#         self._logger = logger

#         self.trajectory = None
        
#         self._nq = plant.num_positions(iiwa_instance)
#         self._nv = plant.num_velocities(iiwa_instance)
#         assert self._nq == self._nv  # iiwa はそうなっているはず

#         self._q_min = np.array([
#             -2.5, -2.5, -2.5, -2.5, 2.5, -2.5, -2.5,
#         ])
#         self._q_max = np.array([
#             2.5,  2.5,  2.5,  2.5,  2.5,  2.5,  2.5,
#         ])
#         self._q_dot_max = np.array([
#             2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2,
#         ])

#         # 1. iiwa_state: [q; v]
#         self.DeclareVectorInputPort("iiwa_state", BasicVector(2 * self._nq))

#         # 2. goal_transform: RigidTransform
#         self.DeclareAbstractInputPort(
#             "goal_transform",
#             AbstractValue.Make(RigidTransform())
#         )

#         # 3. goal_spatial_velocity: SpatialVelocity
#         self.DeclareAbstractInputPort(
#             "goal_spatial_velocity",
#             AbstractValue.Make(SpatialVelocity())
#         )

#         # 4. strike_time: scalar ts
#         self.DeclareVectorInputPort("strike_time", BasicVector(1))

#         # 5. desired_state: [q_des; v_des]
#         self.DeclareVectorOutputPort(
#             "desired_state",
#             BasicVector(2 * self._nq),
#             self.CalcDesiredState,
#         )

#     def CalcDesiredState(self, context: Context, output: BasicVector) -> None:
#         # Prepare plant context
#         plant_context = self._plant.CreateDefaultContext()
#         controller_plant_context = self._controller_plant.CreateDefaultContext()
#         # debug print

#         # Get inputs
#         iiwa_state = self.get_input_port(0).Eval(context)
#         X_WGoal = self.get_input_port(1).Eval(context)
#         V_WGoal = self.get_input_port(2).Eval(context)
#         strike_time = self.get_input_port(3).Eval(context)[0]
#         q_current = iiwa_state[:self._nq]
#         v_current = iiwa_state[self._nq:2*self._nq]

#         # Prepare current translation of the end-effector
#         self._plant.SetPositions(plant_context, self._iiwa, q_current)
#         self._plant.SetVelocities(plant_context, self._iiwa, v_current)
#         ee = self._plant.GetBodyByName("iiwa_link_7")
#         X_WStart = self._plant.EvalBodyPoseInWorld(plant_context, ee)

#         self._controller_plant.SetPositions(controller_plant_context, self._iiwa, q_current)
#         self._controller_plant.SetVelocities(controller_plant_context, self._iiwa, v_current)
#         ee_controller = self._controller_plant.GetBodyByName("iiwa_link_7")
#         ee_frame_controller = ee_controller.body_frame()
#         X_WStart_controller = self._controller_plant.EvalBodyPoseInWorld(controller_plant_context, ee_controller)

#         # Prepare goal configuration velocity using Jacobian inverse
#         J_W = self._plant.CalcJacobianSpatialVelocity(
#             plant_context,
#             JacobianWrtVariable.kV,
#             self.ee_frame,
#             self._normal_offset,
#             self._plant.world_frame(),
#             self._plant.world_frame(),
#         )[:3, :self._nq]  # Take only translational part
#         J_W_pseudo_inv = np.linalg.pinv(J_W)
#         v_desired = J_W_pseudo_inv @ V_WGoal.translational()

#         if np.any(np.isnan(v_desired)) or np.any(np.isinf(v_desired)):
#             q_desired = q_current
#             v_desired = np.zeros(self._nq)
#             desired_state = np.concatenate([q_desired, v_desired])
#             output.SetFromVector(desired_state)
#             return


#         # Use KinematicTrajectoryOptimization to compute desired state
#         if self.trajectory is None:
#             trajopt = KinematicTrajectoryOptimization(
#                 self._nq,
#                 self.N,
#             )
#         else:
#             trajopt = KinematicTrajectoryOptimization(
#                 self.trajectory,
#             )

        
#         # Formulate optimization
#         prog = trajopt.get_mutable_prog()

#         # Add cost to path length
#         trajopt.AddPathLengthCost(1.0)
        
#         #  Add boundary conditions
#         trajopt.AddPositionBounds(
#             self._plant.GetPositionLowerLimits()[:self._nq], self._plant.GetPositionUpperLimits()[:self._nq]
#         )
#         trajopt.AddVelocityBounds(
#             self._plant.GetVelocityLowerLimits()[:self._nv], self._plant.GetVelocityUpperLimits()[:self._nv]
#         )
#         # trajopt.AddPositionBounds(
#         #     self._q_min, self._q_max
#         # )
#         # print("Custom position limits:", self._q_min, self._q_max)
#         # trajopt.AddVelocityBounds(
#         #     -self._q_dot_max, self._q_dot_max
#         # )
#         # print("Custom velocity limits:", -self._q_dot_max, self._q_dot_max)

#         # Add start constraint
#         error = np.ones(3) * 2e-1
#         start_constraint = PositionConstraint(
#             self._controller_plant,
#             self._controller_plant.world_frame(),
#             X_WStart_controller.translation() - error,
#             X_WStart_controller.translation() + error,
#             ee_frame_controller, 
#             [0, 0, 0],
#             controller_plant_context
#         )

#         trajopt.AddPathPositionConstraint(start_constraint, 0.0)

#         prog.AddQuadraticErrorCost(
#             np.eye(self._nq),
#             q_current,
#             trajopt.control_points()[:, 0]
#         )

#         # Add goal constraint
#         goal_constraint = PositionConstraint(
#             self._controller_plant,
#             self._controller_plant.world_frame(),
#             X_WGoal.translation() - error,
#             X_WGoal.translation() + error,
#             ee_frame_controller,
#             [0, 0, 0],
#             controller_plant_context
#         )

#         trajopt.AddPathPositionConstraint(goal_constraint, 1.0)
#         prog.AddQuadraticErrorCost(
#             np.eye(self._nq),
#             q_current,
#             trajopt.control_points()[:, -1]
#         )

#         # start at the current velocity
#         v_error = np.ones(7) * 1e-2
#         trajopt.AddPathVelocityConstraint(
#             v_current - v_error, v_current + v_error, 0.0
#         )

#         # end at the desired velocity
#         trajopt.AddPathVelocityConstraint(
#             v_desired - v_error, v_desired + v_error, 1.0
#         )

#         # Solve optimization
#         if self.trajectory is not None:
#             result = Solve(prog, initial_guess=self.trajectory)
#         else:
#             result = Solve(prog)
        
#         if not result.is_success():
#             q_desired = q_current
#             v_desired = np.zeros(self._nq)
#             desired_state = np.concatenate([q_desired, v_desired])
#             output.SetFromVector(desired_state)
#             print("[WARNING] Trajectory optimization failed.")
#             return
#         print("[INFO] Trajectory optimization succeeded.")
            

#         self.trajectory = trajopt.ReconstructTrajectory(result)

#         # Set Output desired state at current time
#         tau_next = 0.05  # small time step
#         q_desired = self.trajectory.value(tau_next)[:, 0]
#         v_desired = self.trajectory.MakeDerivative(1).value(tau_next)[:, 0]
#         print("Desired q:", q_desired)
#         print("Desired v:", v_desired)
#         desired_state = np.concatenate([q_desired, v_desired])
#         output.SetFromVector(desired_state)

#         if self._meshcat is not None:
#             for i, tau in enumerate(np.linspace(0, 1.0, 20)):
#                 q_i = self.trajectory.value(tau)[:, 0]
#                 self._plant.SetPositions(plant_context, self._iiwa, q_i)
#                 ee_body = self._plant.GetBodyByName("iiwa_link_7")
#                 X_WB = self._plant.EvalBodyPoseInWorld(plant_context, ee_body)
#                  # Visualize end-effector trajectory
#                 self._meshcat.SetObject(
#                     f"trajectory/ee_{i}",
#                     Sphere(0.02),
#                     Rgba(0, 0, 1.0, 0.5),
#                 )
#                 self._meshcat.SetTransform(
#                     f"trajectory/ee_{i}",
#                     X_WB
#                 )

class SimpleOptimizeTrajectory(LeafSystem):

    def __init__(
        self,
        plant: MultibodyPlant,
        iiwa_instance: ModelInstanceIndex,
        controller_plant: MultibodyPlant,
        end_effector_frame_name: str,
        normal_offset_in_ee: np.ndarray,
        horizon: float = 1.5,
        N_ctrl: int = 20,
        num_limit_samples: int = 10,
        max_sqp_iters: int = 5,
        meshcat=None,
        logger=None,
    ):
        super().__init__()

        self._plant = plant
        self._controller_plant = controller_plant
        self._iiwa = iiwa_instance
        self.ee_frame = controller_plant.GetFrameByName(
            end_effector_frame_name, iiwa_instance
        )

        self._offset = np.array(normal_offset_in_ee, float)
        self.horizon = horizon
        self.N = N_ctrl
        self._meshcat = meshcat
        self._logger = logger

        self._nq = plant.num_positions(iiwa_instance)
        self._nv = plant.num_velocities(iiwa_instance)

        # I/O ports (unchanged)
        self.DeclareVectorInputPort("iiwa_state", BasicVector(2 * self._nq))
        self.DeclareAbstractInputPort(
            "goal_transform", AbstractValue.Make(RigidTransform())
        )
        self.DeclareAbstractInputPort(
            "goal_spatial_velocity", AbstractValue.Make(SpatialVelocity())
        )
        self.DeclareVectorInputPort("strike_time", BasicVector(1))
        self.DeclareVectorOutputPort(
            "desired_state",
            BasicVector(2 * self._nq),
            self.CalcDesiredState,
        )

        self._last_traj = None


    # ===============================================================
    # Main function that computes desired state
    # ===============================================================
    def CalcDesiredState(self, context, output):
        # DEBUG
        debug = True
        if debug:
            nominal_q = np.ones(self._nq) * 0.6
            nominal_v = np.zeros(self._nq)
            output.SetFromVector(np.concatenate([nominal_q, nominal_v]))

            # visualize end-effector at nominal pose
            plant_context = self._plant.CreateDefaultContext()
            self._plant.SetPositions(plant_context, self._iiwa, nominal_q)
            ee_body = self._plant.GetBodyByName("iiwa_link_7")
            X_WB = self._plant.EvalBodyPoseInWorld(plant_context, ee_body)
            if self._meshcat is not None:
                self._meshcat.SetObject(
                    "nominal_ee",
                    Sphere(0.02),
                    Rgba(1.0, 0, 0, 0.5),
                )
                self._meshcat.SetTransform(
                    "nominal_ee",
                    X_WB
                )
            return


        # --------- Read inputs ----------
        iiwa_state = self.get_input_port(0).Eval(context)
        X_goal = self.get_input_port(1).Eval(context)
        V_goal = self.get_input_port(2).Eval(context)
        strike_time = float(self.get_input_port(3).Eval(context)[0])

        if strike_time < 1e-4 or strike_time > self.horizon:
            # Out of bounds: return current state
            q0 = iiwa_state[:self._nq]
            v0 = iiwa_state[self._nq:]
            output.SetFromVector(np.concatenate([q0, v0]))
            return

         # Current state

        q0 = iiwa_state[:self._nq]
        v0 = iiwa_state[self._nq:]

        # --------- Compute v_desired (IK from Jacobian) ----------
        plant_context = self._plant.CreateDefaultContext()
        self._plant.SetPositions(plant_context, self._iiwa, q0)
        self._plant.SetVelocities(plant_context, self._iiwa, v0)

        J = self._plant.CalcJacobianSpatialVelocity(
            plant_context,
            JacobianWrtVariable.kV,
            self.ee_frame,
            self._offset,
            self._plant.world_frame(),
            self._plant.world_frame(),
        )[:3, :self._nq]

        v_des = np.linalg.pinv(J) @ V_goal.translational()

        if np.any(~np.isfinite(v_des)):
            # Fail-safe
            output.SetFromVector(np.concatenate([q0, np.zeros_like(v0)]))
            return

        # ======================================================
        # Trajectory optimization
        # ======================================================
        if self._last_traj is None:
            trajopt = KinematicTrajectoryOptimization(self._nq, self.N)
        else:
            trajopt = KinematicTrajectoryOptimization(self._last_traj)
        prog = trajopt.get_mutable_prog()

        # cost
        trajopt.AddPathLengthCost(1.0)

        # Joint limits
        trajopt.AddPositionBounds(
            self._plant.GetPositionLowerLimits()[: self._nq],
            self._plant.GetPositionUpperLimits()[: self._nq],
        )
        velocity_limits = np.ones(self._nq) * 0.05  # iiwa velocity limits
        trajopt.AddVelocityBounds(
            -velocity_limits,
            velocity_limits,
        )

        # ------------------------------------------------------
        # Start/Goal constraints in task space
        # ------------------------------------------------------

        ctrl_context = self._controller_plant.CreateDefaultContext()
        self._controller_plant.SetPositions(ctrl_context, self._iiwa, q0)

        X_start = self._controller_plant.EvalBodyPoseInWorld(
            ctrl_context, 
            self._controller_plant.GetBodyByName("iiwa_link_7")
        ) 

        eps = np.ones(3) * 1e-2  # small tolerance

        # Start position
        cstart = PositionConstraint(
            self._controller_plant,
            self._controller_plant.world_frame(),
            X_start.translation(),
            X_start.translation(),
            self.ee_frame,
            np.zeros(3),
            ctrl_context
        )
        trajopt.AddPathPositionConstraint(cstart, 0.0)

        # Goal position
        cgoal = PositionConstraint(
            self._controller_plant,
            self._controller_plant.world_frame(),
            X_goal.translation(),
            X_goal.translation(),
            self.ee_frame,
            np.zeros(3),
            ctrl_context
        )
        trajopt.AddPathPositionConstraint(cgoal, 1.0)

        # Start velocity
        dv = np.ones(self._nq) * 1e-3
        trajopt.AddPathVelocityConstraint(v0 - dv, v0 + dv, 0.0)

        # End velocity
        #trajopt.AddPathVelocityConstraint(v_des - dv, v_des + dv, 1.0)
        zero_vec = np.zeros_like(v_des)
        trajopt.AddPathVelocityConstraint(zero_vec, zero_vec, 1.0)

        # # Start Orientation
        # orientation_start = OrientationConstraint(
        #     self._controller_plant,
        #     self._controller_plant.world_frame(),
        #     X_start.rotation(),
        #     self.ee_frame,
        #     RotationMatrix().Identity(),
        #     0.1,
        #     ctrl_context
        # )
        # trajopt.AddPathPositionConstraint(orientation_start, 0.0)

        # # Goal Orientation
        # orientation_goal = OrientationConstraint(
        #     self._controller_plant,
        #     self._controller_plant.world_frame(),
        #     X_goal.rotation(),
        #     self.ee_frame,
        #     RotationMatrix().Identity(),
        #     0.1,
        #     ctrl_context
        # )
        # trajopt.AddPathPositionConstraint(orientation_goal, 1.0)
        


        # ------------------------------------------------------
        # Initial guess
        # ------------------------------------------------------
        # linear interpolation between q0 and IK goal
        q_goal_guess = q0 + 0.5 * (v_des * strike_time)

        if self._last_traj is None:
            for k in range(self.N):
                alpha = k / (self.N - 1)
                guess = (1 - alpha) * q0 + alpha * q_goal_guess
                prog.SetInitialGuess(trajopt.control_points()[:, k], guess)

        # ------------------------------------------------------
        # Solve
        # ------------------------------------------------------
        result = Solve(prog)

        if not result.is_success():
            # print("[WARN] traj opt failed")
            output.SetFromVector(np.concatenate([q0, np.zeros_like(v0)]))
            return

        traj = trajopt.ReconstructTrajectory(result)
        self._last_traj = traj

        # Return small-step future
        if strike_time > 0:
            t = min(0.08 / strike_time, 1.0)
        else:
            t = 1.0
        qd = traj.value(t).flatten()
        vd = traj.MakeDerivative(1).value(t).flatten()

        #output.SetFromVector(np.concatenate([qd, vd]))
        nominal_q = np.ones_like(qd)*0.6
        nominal_v = np.zeros_like(vd)
        output.SetFromVector(np.concatenate([nominal_q, nominal_v]))

        if self._meshcat:
            X = self._plant.SetPositions(plant_context, self._iiwa, nominal_q)
            X = self._plant.EvalBodyPoseInWorld(
                plant_context, 
                self._plant.GetBodyByName("iiwa_link_7")
            )
            self._meshcat.SetObject("nominal", Sphere(0.03), Rgba(0,0.5,0,0.5))
            self._meshcat.SetTransform("nominal", X)


        # visualization
        if self._meshcat:
            for i, s in enumerate(np.linspace(0, 1, 20)):
                qi = traj.value(s).flatten()
                self._plant.SetPositions(plant_context, self._iiwa, qi)
                X = self._plant.EvalBodyPoseInWorld(
                    plant_context, 
                    self._plant.GetBodyByName("iiwa_link_7")
                )
                self._meshcat.SetObject(f"traj/{i}", Sphere(0.02), Rgba(0,0,1,0.5))
                self._meshcat.SetTransform(f"traj/{i}", X)



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
        horizon: float = 1.5,
        N_ctrl: int = 8,
        num_limit_samples: int = 10,
        W_a: float = 1.0,
        W_r: float = 1e-2,
        W_n: float = 1.0,
        max_sqp_iters: int = 5,
        meshcat = None,
        logger = None,
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
        self._meshcat = meshcat
        self._logger = logger

        # 関節数とリミット取得
        self._nq = plant.num_positions(iiwa_instance)
        self._nv = plant.num_velocities(iiwa_instance)
        assert self._nq == self._nv  # iiwa はそうなっているはず

        self._q_min = np.array([
            -3.1, -3.1, -3.1, -3.1, 3.1, -3.1, -3.1,
        ])
        self._q_max = np.array([
            3.1,  3.1,  3.1,  3.1,  3.1,  3.1,  3.1,
        ])
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

        # 初期条件 (4b),(4c)
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, B0) - q0, np.zeros(nq))
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, dB0) - v0, np.zeros(nq))

        # slack とのリンク (5a),(5b)
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, Bs) - q_s, np.zeros(nq))
        prog.AddLinearEqualityConstraint(bezier(q_ctrl, dBs) - qsdot, np.zeros(nq))

        # # joint limit (4d): 時間サンプリング
        if self._num_limit_samples > 0:
            taus = np.linspace(0.0, 1.0, self._num_limit_samples + 1)[1:]  # 0 は除外
            for tau in taus:
                Bt = self._bezier_basis(tau)
                qt = bezier(q_ctrl, Bt)
                for i in range(nq):
                    prog.AddLinearConstraint(qt[i] >= self._q_min[i])
                    prog.AddLinearConstraint(qt[i] <= self._q_max[i])

        # EE 関連制約 (4f),(4g),(4h) を q_lin で線形化
        context = self._plant.CreateDefaultContext()
        if len(q_lin) != self._plant.num_positions(self._iiwa):
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
        t_now = min(0.1, T)   # 1e-2[s] だけ先
        tau_now = t_now / T
        B_now = self._bezier_basis(tau_now)
        dB_now = self._bezier_basis_derivative(tau_now)

        q_des = q_ctrl @ B_now
        v_des = q_ctrl @ dB_now

        # Debug: visualize trajectory in Meshcat
        for i, tau in enumerate(np.linspace(0, 5, 40)):
            B = self._bezier_basis(tau)
            dB = self._bezier_basis_derivative(tau)
            q_t = q_ctrl @ B
            v_t = q_ctrl @ dB

            context_tmp = self._plant.CreateDefaultContext()
            self._plant.SetPositions(context_tmp, self._iiwa, q_t)
            ee = self._plant.GetBodyByName("iiwa_link_7")
            X_WB = self._plant.EvalBodyPoseInWorld(context_tmp, ee)
            if self._meshcat is not None:
                # Draw spheres along the trajectory in Meshcat
                self._meshcat.SetObject(
                    f"bezier_trajectory/sphere_{i}",
                    shape = Sphere(0.02),
                    rgba=Rgba(0.0, 1.0, 0.0, 0.5)
                )
                self._meshcat.SetTransform(
                    f"bezier_trajectory/sphere_{i}",
                    X_WB
                )

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
        # print(f"ts={ts}, goal_T={goal_T.translation()}")

        q_des, v_des, _ = self._solve_mpc(q0, v0, ts, goal_T, goal_V)

        y = np.zeros(2 * self._nq)
        y[: self._nq] = q_des
        y[self._nq :] = v_des
        output.SetFromVector(y)
        if self._logger is not None:
            self._logger.Log(f"DesiredStateInTrajectoryOptim {y}")

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

