import numpy as np
from pydrake.systems.framework import LeafSystem, Context, BasicVector


class SimpleController(LeafSystem):
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

