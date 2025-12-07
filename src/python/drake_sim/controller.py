import numpy as np
from pydrake.systems.framework import LeafSystem, Context, BasicVector
from pydrake.all import (
    RigidTransform, SpatialVelocity, AbstractValue,
    MathematicalProgram, Solve, eq, ge, le, MultibodyPlant,
    JacobianWrtVariable, MultibodyForces
    )


class SimpleController(LeafSystem):
    """
    A simple PD controller for a 7-DOF robotic arm.
    """
    def __init__(self, k_p: float=90.0, k_i: float=500.0, k_d: float=1000000.0, q_desired: np.ndarray=None):
        super().__init__()
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.q_desired = q_desired if q_desired is not None else np.zeros(7)
        self.DeclareVectorInputPort(name="iiwa_state", size=14)
        self.input_port = self.get_input_port(0)
        self.DeclareVectorOutputPort(name="iiwa_torque", size=7, calc=self.ComputeTorque)
        self.output_port = self.get_output_port(0)
        self.previous_q = None
        self.integral_error = np.zeros(7)
        self.previous_time = 0.0

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        q = self.input_port.Eval(context)[:7]
        current_time = context.get_time()
        dt = current_time - self.previous_time
        if dt > 0:
            self.integral_error += (self.q_desired - q) * dt
        torque = self.k_p * (self.q_desired - q) + self.k_i * self.integral_error + self.k_d * (self.previous_q - q) if self.previous_q is not None else np.zeros(7)
        self.previous_q = q.copy()
        self.previous_time = current_time
        output.SetFromVector(torque)

class WSGController(LeafSystem):
    """
    A simple PD controller for a Weiss WSG gripper.
    """
    def __init__(self, position_desired: np.ndarray, k_p: float=8.0, k_i: float=20.0, k_d: float=100.0, max_torque: float = 100.0):
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

class OptimizeTrajectory(LeafSystem):
    """
    Calculates optimal trajectory in the parameter space of a 7-DOF robotic arm.
    Trajectory is optimized in the internal function and the current desired joint configurations and velocities are outputted.
    inputs:
        - goal_transform(RigidTransform): RigidTransform representing desired end-effector pose
        - goal_spatial_velocity(SpatialVelocity): SpatialVelocity representing desired end-effector velocity
    outputs:
        - q_desired (7x1 Vector): Desired joint positions
        - qd_desired (7x1 Vector): Desired joint velocities
    """
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()

        # Inputs
        self.DeclareAbstractInputPort(
            name="goal_transform",
            model_value=AbstractValue.Make(RigidTransform())
        )
        self.DeclareAbstractInputPort(
            name="goal_spatial_velocity",
            model_value=AbstractValue.Make(SpatialVelocity())
        )

        # Outputs
        self.DeclareVectorOutputPort(
            name="desired_state",
            size=14,
            calc=self.ComputeDesiredState
        )
        self.DeclareVectorOutputPort(
            name="qdd_desired",
            size=7,
            calc=self.CalcQddDesired
        )

        # Internal storage
        self._q_desired = np.zeros(7)
        self._qd_desired = np.zeros(7)
        self._qdd_desired = np.zeros(7)

    def solve_trajectory_optimization(self, context: Context):
        """
        Solve a simple quadratic program to find joint positions and velocities that
        achieve the desired end-effector pose and spatial velocity.
        """
        goal_X = self.get_input_port(0).Eval(context)
        goal_V = self.get_input_port(1).Eval(context)

        prog = MathematicalProgram()
        q = prog.NewContinuousVariables(7, "q")
        qd = prog.NewContinuousVariables(7, "qd")

        # Update context with candidate q, qd (symbolic)
        print("num_positions:", self.plant.num_positions())
        print("num_velocities:", self.plant.num_velocities())
        self.plant.SetPositions(self.plant_context, np.zeros(16))  # initial guess
        self.plant.SetVelocities(self.plant_context, np.zeros(15))

        # End-effector frame
        ee_frame = self.plant.GetBodyByName("iiwa_link_7").body_frame()
        world_frame = self.plant.world_frame()

        # Jacobian at nominal position (linearization)
        J = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            ee_frame,
            np.zeros(3),
            world_frame,
            world_frame
        )[:6, :7]  # 6x7 Jacobian

        # Linear cost: stay near current pose
        prog.AddQuadraticCost(q @ q)  # minimize magnitude of q
        prog.AddQuadraticCost(qd @ qd)  # minimize velocities

        # Constraints: end-effector position & velocity
        # Here, linearize position via Jacobian (6x7) to approximate pose error
        ee_pos = goal_X.translation()  # 3D target
        current_pos = self.plant.CalcRelativeTransform(
            self.plant_context, frame_A=world_frame, frame_B=ee_frame
        ).translation()
        pos_error = ee_pos - current_pos
        prog.AddLinearConstraint(J[:3, :] @ q == pos_error)  # position
        prog.AddLinearConstraint(J[3:, :] @ qd == goal_V.rotational())  # angular velocity

        # Solve
        result = Solve(prog)
        if not result.is_success():
            raise RuntimeError("Trajectory optimization failed.")

        self._q_desired = result.GetSolution(q)
        self._qd_desired = result.GetSolution(qd)

        # For simplicity, desired acceleration is PD towards goal
        k_p = 100.0
        k_d = 20.0
        self._qdd_desired = k_p * (self._q_desired - np.zeros(7)) + k_d * (self._qd_desired - np.zeros(7))

    def ComputeDesiredState(self, context: Context, output: BasicVector) -> None:
        self.solve_trajectory_optimization(context)
        desired_state = np.concatenate([self._q_desired, self._qd_desired])
        output.SetFromVector(desired_state)

    def CalcQddDesired(self, context: Context, output: BasicVector) -> None:
        output.SetFromVector(self._qdd_desired)


    
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
        self.plant.SetPositions(self.plant_context, q)
        self.plant.SetVelocities(self.plant_context, qd)

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
        b_dyn = tau_g + f_ext - C_matrix
        prog.AddLinearEqualityConstraint(A_dyn, np.concatenate([q_dd, tau]), b_dyn)

        # Solve
        result = Solve(prog, initial_guess=self._initial_guess)
        if not result.is_success():
            raise RuntimeError("Optimization failed to find a solution.")

        optimal_tau = result.GetSolution(tau)
        output.SetFromVector(optimal_tau)
        self._initial_guess = np.concatenate([result.GetSolution(q_dd), optimal_tau])


