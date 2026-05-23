"""Discrete-time RC thermal model.

Implements a first-order resistance-capacitance (RC) building model:

    C * dT_in/dt = (T_out - T_in) / R
                  + eta_hvac * P_hvac
                  + Q_int

Discretised with the exact zero-order hold (ZOH) solution so that
large time steps (e.g. 15 min) remain numerically stable.

Reference:
    Equation (3) in the project theoretical section; see also
    the RC thermal modelling literature (TU Delft report, HAL report).
"""

from __future__ import annotations

import math
import numpy as np
from smart_hvac.core.parameters import Parameters


# ------------------------------------------------------------------ #
#  Core update step                                                    #
# ------------------------------------------------------------------ #

def step_rc(
    T_in: float,
    T_out: float,
    P_hvac: float,
    Q_int: float,
    params: Parameters,
) -> float:
    """Advance the indoor temperature by one time step dt.

    Uses the exact ZOH (zero-order hold) analytical solution of the
    first-order linear ODE over [t, t+dt], assuming all inputs are
    constant within the interval.

    Parameters
    ----------
    T_in   : current indoor temperature [°C]
    T_out  : outdoor temperature during this step [°C]
    P_hvac : HVAC power applied during this step [W]
    Q_int  : internal heat gains during this step [W]
    params : Parameters instance

    Returns
    -------
    T_in_next : indoor temperature at t + dt [°C]
    """
    alpha = math.exp(-params.dt / params.tau)         # decay factor ∈ (0, 1)

    # Steady-state indoor temperature given constant inputs:
    #   T_ss = T_out + R * (eta_hvac * P_hvac + Q_int)
    T_ss = T_out + params.R * (params.eta_hvac * P_hvac + Q_int)

    # ZOH exact solution:  T_in(t+dt) = T_ss + (T_in(t) - T_ss) * alpha
    T_in_next = T_ss + (T_in - T_ss) * alpha
    return float(T_in_next)


# ------------------------------------------------------------------ #
#  Weather generators                                                  #
# ------------------------------------------------------------------ #

def sinusoidal_temperature(
    step: int,
    params: Parameters,
    T_mean: float,
    rng: np.random.Generator,
) -> float:
    """Return outdoor temperature for a given simulation step.

    T_out(t) = T_mean + A * sin(2π * t/24h + phase_offset) + noise

    The daily minimum is around 04:00; maximum around 14:00.

    Parameters
    ----------
    step    : current simulation step index (0-based)
    params  : Parameters instance (dt, T_amplitude, T_noise_std)
    T_mean  : mean outdoor temperature for the day [°C]
    rng     : numpy random Generator for reproducibility
    """
    t_hours = (step * params.dt) / 3600.0      # fractional hour of day
    phase = 2.0 * math.pi * t_hours / 24.0

    # Shift so peak is at ~14:00 (14/24 * 2π - π/2)
    T_out = T_mean + params.T_amplitude * math.sin(phase - math.pi / 2)
    T_out += float(rng.normal(0.0, params.T_noise_std))
    return T_out


def load_weather_csv(path: str, dt_s: float) -> np.ndarray:
    """Load outdoor temperature time series from a CSV file.

    The CSV must have at least a column named 'T_out' with temperature
    values in °C at any uniform time resolution.  The series is
    resampled (nearest-neighbour) to the simulation dt.

    Parameters
    ----------
    path  : path to CSV file
    dt_s  : simulation time step in seconds

    Returns
    -------
    T_out_series : 1-D numpy array of temperatures at simulation dt
    """
    import pandas as pd

    df = pd.read_csv(path)
    if "T_out" not in df.columns:
        raise ValueError("CSV must contain a column named 'T_out'.")

    raw = df["T_out"].to_numpy(dtype=float)
    # Infer original resolution and resample if needed
    # (simple approach: assume 1-hour resolution → 4 steps per hour at 15 min)
    raw_dt_s = 3600.0  # assumed default; user can override
    if "dt_seconds" in df.columns:
        raw_dt_s = float(df["dt_seconds"].iloc[0])

    ratio = raw_dt_s / dt_s
    if abs(ratio - round(ratio)) < 1e-6:
        ratio = int(round(ratio))
        T_out_series = np.repeat(raw, ratio) if ratio > 1 else raw[::int(1 / ratio)]
    else:
        # General resampling via interpolation
        x_raw = np.arange(len(raw)) * raw_dt_s
        x_new = np.arange(0, x_raw[-1], dt_s)
        T_out_series = np.interp(x_new, x_raw, raw)

    return T_out_series.astype(np.float32)


# ------------------------------------------------------------------ #
#  Occupancy schedule generators                                       #
# ------------------------------------------------------------------ #

def occupancy_schedule(
    episode_steps: int,
    dt_s: float,
    profile: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a binary occupancy array for one episode.

    Parameters
    ----------
    episode_steps : number of simulation steps in the episode
    dt_s          : step duration in seconds
    profile       : one of 'office', 'residential', 'always_on'
    rng           : optional RNG for stochastic occupancy

    Returns
    -------
    occ : int8 array of shape (episode_steps,) with values 0 or 1
    """
    occ = np.zeros(episode_steps, dtype=np.int8)
    steps_per_hour = int(3600.0 / dt_s)

    if profile == "always_on":
        occ[:] = 1

    elif profile == "office":
        # Occupied 08:00–18:00
        start = 8 * steps_per_hour
        end = 18 * steps_per_hour
        occ[start:end] = 1

    elif profile == "residential":
        # Morning 07:00–09:00
        occ[7 * steps_per_hour: 9 * steps_per_hour] = 1
        # Evening 18:00–23:00
        occ[18 * steps_per_hour: 23 * steps_per_hour] = 1

    else:
        raise ValueError(f"Unknown occupancy profile: {profile!r}")

    # Optional random absences (~10 % of occupied steps)
    if rng is not None:
        mask = occ == 1
        absences = rng.random(mask.sum()) < 0.10
        occ[np.where(mask)[0][absences]] = 0

    return occ