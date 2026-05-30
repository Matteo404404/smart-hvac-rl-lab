"""Global Parameters dataclass for the smart-hvac-rl-lab project."""

from dataclasses import dataclass, field


@dataclass
class Parameters:
    # ── Thermal RC model ────────────────────────────────────────────────────
    C: float = 1_800_000.0   # J/K  – thermal capacitance (air + light mass)
    R: float = 0.005          # K/W  – thermal resistance wall/window
    eta_hvac: float = 1.0     # dimensionless – HVAC efficiency factor
    P_max: float = 6_000.0    # W    – maximum HVAC heating power

    # ── Internal gains ──────────────────────────────────────────────────────
    Q_occ: float = 150.0      # W    – heat gain per occupant
    Q_noise_std: float = 20.0 # W    – stochastic noise on internal gains

    # ── Reward weights ──────────────────────────────────────────────────────
    lambda_c: float = 1.0     # comfort penalty weight
    lambda_e: float = 0.0001   # energy penalty weight  (per Watt)
    lambda_s: float = 0.05
    lambda_b: float = 0.5    # in-band comfort bonus (reward per timestep inside comfort band)

    # ── Comfort band ────────────────────────────────────────────────────────
    T_min: float = 21.0       # °C – lower comfort bound
    T_max: float = 23.0       # °C – upper comfort bound

    # ── Simulation ──────────────────────────────────────────────────────────
    dt: float = 900.0         # s   – timestep (15 min)
    episode_steps: int = 96   # steps per episode (24 h at 15-min resolution)

    # ── Weather ─────────────────────────────────────────────────────────────
    weather_profile: str = "sinusoidal"   # "sinusoidal" | "csv"
    weather_csv_path: str = ""            # path to optional CSV file
    T_out_noise_std: float = 0.5          # °C – outdoor temp noise std

    # ── Occupancy ───────────────────────────────────────────────────────────
    occupancy_profile: str = "office"     # "office" | "residential" | "always"

    # ── Day-type weather parameters (filled by WeatherProcess) ──────────────
    day_type_params: dict = field(default_factory=lambda: {
        "cold":  {"T_mean": 2.0,  "T_amp": 4.0},
        "mild":  {"T_mean": 12.0, "T_amp": 6.0},
        "hot":   {"T_mean": 28.0, "T_amp": 5.0},
    })
