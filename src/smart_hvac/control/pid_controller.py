"""Discrete-time PID controller baseline for HVAC."""

from __future__ import annotations
import numpy as np
from smart_hvac.core.parameters import Parameters


class PIDController:
    """Discrete PID with anti-windup and output clipping to [0, 1].

    Setpoint: midpoint of comfort band = (T_min + T_max) / 2.

    Parameters
    ----------
    Kp, Ki, Kd : PID gains
    dt         : timestep [s]
    """

    def __init__(
        self,
        params: Parameters | None = None,
        Kp: float = 0.15,
        Ki: float = 0.008,
        Kd: float = 0.5,
    ):
        self.params = params or Parameters()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = self.params.dt
        self._T_set = (self.params.T_min + self.params.T_max) / 2.0
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return action given observation vector."""
        T_in = float(obs[0])
        error = self._T_set - T_in

        self._integral += error * self.dt
        # Anti-windup: clamp integral contribution
        self._integral = float(np.clip(
            self._integral, -1.0 / (self.Ki + 1e-9), 1.0 / (self.Ki + 1e-9)
        ))

        derivative = (error - self._prev_error) / self.dt
        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        self._prev_error = error

        u_clipped = float(np.clip(u, 0.0, 1.0))
        return np.array([u_clipped], dtype=np.float32)
