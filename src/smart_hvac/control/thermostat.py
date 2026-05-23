"""On/off hysteresis thermostat baseline.

Switches heating ON (u=1) when T_in drops below T_set_low,
and OFF (u=0) when T_in rises above T_set_high.
Between the two thresholds the previous output is held (hysteresis).
"""

from __future__ import annotations
from smart_hvac.core.parameters import Parameters


class Thermostat:
    """Simple bang-bang thermostat controller.

    Parameters
    ----------
    params      : Parameters instance (T_min, T_max used as band)
    T_set_low   : turn heating ON below this temperature [°C]
    T_set_high  : turn heating OFF above this temperature [°C]
    """

    def __init__(
        self,
        params: Parameters | None = None,
        T_set_low: float | None = None,
        T_set_high: float | None = None,
    ) -> None:
        self.params = params or Parameters()
        self.T_set_low = T_set_low if T_set_low is not None else self.params.T_min
        self.T_set_high = T_set_high if T_set_high is not None else self.params.T_max
        self._u: float = 0.0  # previous output (for hysteresis)

    def reset(self) -> None:
        """Reset internal state (call at the start of each episode)."""
        self._u = 0.0

    def act(self, T_in: float) -> float:
        """Compute normalised HVAC power command.

        Parameters
        ----------
        T_in : current indoor temperature [°C]

        Returns
        -------
        u : normalised power in [0, 1]
        """
        if T_in < self.T_set_low:
            self._u = 1.0
        elif T_in > self.T_set_high:
            self._u = 0.0
        # else: hold previous output (hysteresis band)
        return self._u