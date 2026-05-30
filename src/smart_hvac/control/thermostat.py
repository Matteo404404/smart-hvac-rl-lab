"""Simple on/off hysteresis thermostat baseline."""

from __future__ import annotations
import numpy as np
from smart_hvac.core.parameters import Parameters


class Thermostat:
    """Bang-bang controller with hysteresis around the comfort band.

    If T_in < T_min  → heat at full power (u = 1).
    If T_in > T_max  → heating off     (u = 0).
    Otherwise        → hold previous action.
    """

    def __init__(self, params: Parameters | None = None):
        self.params = params or Parameters()
        self._u: float = 0.0

    def reset(self) -> None:
        self._u = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return action given observation vector [T_in, T_out, sin, cos, occ, u_prev]."""
        T_in = float(obs[0])
        if T_in < self.params.T_min:
            self._u = 1.0
        elif T_in > self.params.T_max:
            self._u = 0.0
        return np.array([self._u], dtype=np.float32)
