"""Central Parameters dataclass.

All physical constants, reward weights, and simulation settings
live here. Pass a single Parameters instance everywhere – never
use loose magic numbers in the code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Parameters:
    # ------------------------------------------------------------------ #
    #  Physical / RC model                                                 #
    # ------------------------------------------------------------------ #
    C: float = 1_500_000.0      # Thermal capacitance [J/K]  (~small room)
    R: float = 0.005            # Thermal resistance  [K/W]  (~well-insulated wall)
    eta_hvac: float = 1.0       # HVAC efficiency coefficient [dimensionless]
    P_max: float = 3_000.0      # Max HVAC power [W]
    Q_occ: float = 120.0        # Internal gains when occupied [W]  (~1 person + PC)
    Q_noise_std: float = 10.0   # Std-dev of stochastic internal gains [W]

    # ------------------------------------------------------------------ #
    #  Comfort band                                                        #
    # ------------------------------------------------------------------ #
    T_min: float = 21.0         # Lower comfort bound [°C]
    T_max: float = 23.0         # Upper comfort bound [°C]
    T_set: float = 22.0         # Nominal setpoint (used by thermostat/PID) [°C]

    # ------------------------------------------------------------------ #
    #  Reward weights  (see Eq. 5 in the theoretical section)             #
    # ------------------------------------------------------------------ #
    lambda_c: float = 1.0       # Comfort penalty weight
    lambda_e: float = 1e-4      # Energy penalty weight  (normalise vs comfort)
    lambda_s: float = 0.01      # Actuator smoothness penalty weight

    # ------------------------------------------------------------------ #
    #  Simulation / episode                                                #
    # ------------------------------------------------------------------ #
    dt: float = 900.0           # Time step [s]  (15 minutes)
    episode_steps: int = 96     # Steps per episode  (96 × 15 min = 24 h)

    # ------------------------------------------------------------------ #
    #  Weather profile                                                     #
    # ------------------------------------------------------------------ #
    weather_profile: Literal["sinusoidal", "csv"] = "sinusoidal"
    weather_csv_path: str | None = None  # path if weather_profile == "csv"

    # Sinusoidal weather parameters (cold / mild / hot day types)
    T_mean_cold: float = 2.0
    T_mean_mild: float = 12.0
    T_mean_hot: float = 28.0
    T_amplitude: float = 6.0    # Diurnal swing [K]
    T_noise_std: float = 0.5    # Outdoor temp noise std [K]

    # ------------------------------------------------------------------ #
    #  Occupancy                                                           #
    # ------------------------------------------------------------------ #
    occupancy_profile: Literal["office", "residential", "always_on"] = "office"
    # office:      occupied 08:00–18:00 (steps 32–72 in a 15-min grid)
    # residential: occupied 07:00–09:00 and 18:00–23:00
    # always_on:   always occupied

    # ------------------------------------------------------------------ #
    #  PID controller defaults (used in control/pid_controller.py)        #
    # ------------------------------------------------------------------ #
    pid_Kp: float = 0.5
    pid_Ki: float = 0.01
    pid_Kd: float = 0.05

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #
    @property
    def tau(self) -> float:
        """RC time constant [s]."""
        return self.R * self.C

    @property
    def episode_duration_h(self) -> float:
        """Episode duration in hours."""
        return self.episode_steps * self.dt / 3600.0

    def comfort_deviation(self, T_in: float) -> float:
        """Signed deviation from comfort band (0 if inside band)."""
        if T_in < self.T_min:
            return self.T_min - T_in
        if T_in > self.T_max:
            return T_in - self.T_max
        return 0.0

    def __post_init__(self) -> None:
        if self.weather_profile == "csv" and self.weather_csv_path is None:
            raise ValueError(
                "weather_csv_path must be set when weather_profile='csv'."
            )
        if self.T_min >= self.T_max:
            raise ValueError("T_min must be strictly less than T_max.")
        if self.dt <= 0:
            raise ValueError("Time step dt must be positive.")