"""PID controller baseline.

Tracks a temperature setpoint using proportional–integral–derivative
control, with anti-windup clamping and output clipping to [0, 1].

The control law (discrete form, Eq. 6 in the theoretical section):
    u_t = Kp * e_t  +  Ki * sum(e_τ * dt)  +  Kd * (e_t - e_{t-1}) / dt
"""

from __future__ import annotations
from smart_hvac.core.parameters import Parameters


class PIDController:
    """Discrete PID controller for HVAC temperature tracking.

    Parameters
    ----------
    params   : Parameters instance (pid_Kp, pid_Ki, pid_Kd, T_set, dt)
    setpoint : override for the nominal temperature setpoint [°C]
    """

    def __init__(
        self,
        params: Parameters | None = None,
        setpoint: float | None = None,
    ) -> None:
        self.params = params or Parameters()
        self.setpoint = setpoint if setpoint is not None else self.params.T_set
        self._integral: float = 0.0
        self._e_prev: float = 0.0
        # Anti-windup limits (in °C · s)
        self._integral_max: float = 50.0

    def reset(self) -> None:
        """Reset integrator and previous error (call at episode start)."""
        self._integral = 0.0
        self._e_prev = 0.0

    def act(self, T_in: float) -> float:
        """Compute normalised HVAC power command.

        Parameters
        ----------
        T_in : current indoor temperature [°C]

        Returns
        -------
        u : normalised power in [0, 1]
        """
        e = self.setpoint - T_in
        self._integral = max(
            -self._integral_max,
            min(self._integral_max, self._integral + e * self.params.dt),
        )
        derivative = (e - self._e_prev) / self.params.dt
        self._e_prev = e

        u_raw = (
            self.params.pid_Kp * e
            + self.params.pid_Ki * self._integral
            + self.params.pid_Kd * derivative
        )
        return float(max(0.0, min(1.0, u_raw)))