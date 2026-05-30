"""Discrete-time RC thermal model and weather/occupancy generators.

References
----------
- Eq. (3) in project theoretical section: exact ZOH discretisation of a
  first-order linear ODE with constant input over each interval.
"""

from __future__ import annotations

import math
import numpy as np
from smart_hvac.core.parameters import Parameters


# ── Core thermal update ──────────────────────────────────────────────────────

def step_rc(
    T_in: float,
    T_out: float,
    P_hvac: float,
    Q_int: float,
    params: Parameters,
) -> float:
    """Zero-order-hold exact solution of the first-order RC ODE.

    dT_in/dt = (T_out - T_in) / (R*C) + (eta * P_hvac + Q_int) / C

    Parameters
    ----------
    T_in   : current indoor temperature [°C]
    T_out  : outdoor temperature [°C]
    P_hvac : HVAC power [W]
    Q_int  : internal heat gains [W]
    params : Parameters dataclass

    Returns
    -------
    T_in_next : indoor temperature at next timestep [°C]
    """
    R, C, eta = params.R, params.C, params.eta_hvac
    dt = params.dt

    tau = R * C                                    # time constant [s]
    alpha = math.exp(-dt / tau)                    # decay factor
    T_ss = T_out + R * (eta * P_hvac + Q_int)      # steady-state temperature

    return T_ss + (T_in - T_ss) * alpha


# ── Weather process ──────────────────────────────────────────────────────────

class SinusoidalWeather:
    """Synthetic sinusoidal daily outdoor temperature profile with noise."""

    def __init__(self, params: Parameters, rng: np.random.Generator):
        self.params = params
        self.rng = rng
        self._T_mean: float = 10.0
        self._T_amp: float = 5.0
        self._phase: float = -math.pi / 2   # minimum at midnight

    def reset(self, day_type: str = "mild") -> None:
        dp = self.params.day_type_params[day_type]
        self._T_mean = dp["T_mean"]
        self._T_amp = dp["T_amp"]

    def get(self, step: int) -> float:
        """Return outdoor temperature at simulation step k."""
        t_hours = step * self.params.dt / 3600.0
        T_det = self._T_mean + self._T_amp * math.sin(
            2 * math.pi * t_hours / 24.0 + self._phase
        )
        noise = self.rng.normal(0.0, self.params.T_out_noise_std)
        return T_det + noise


# ── Occupancy process ────────────────────────────────────────────────────────

def occupancy_flag(step: int, profile: str, dt: float = 900.0) -> int:
    """Return 1 if occupied, 0 otherwise.

    Parameters
    ----------
    step    : current simulation step
    profile : "office" | "residential" | "always"
    dt      : timestep in seconds
    """
    t_hours = (step * dt / 3600.0) % 24.0

    if profile == "always":
        return 1
    elif profile == "office":
        return int(8.0 <= t_hours < 18.0)
    elif profile == "residential":
        return int(not (9.0 <= t_hours < 17.0))
    else:
        return 1
