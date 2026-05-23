"""Gymnasium-compatible HVAC environment.

Wraps the RC thermal model and weather / occupancy generators into a
standard Gymnasium Env that can be plugged directly into any
Stable-Baselines3 (or other) RL training loop.

Observation vector (6 dims)
---------------------------
  0  T_in          indoor temperature          [°C]       → normalised to [0,1]
  1  T_out         outdoor temperature         [°C]       → normalised to [0,1]
  2  sin_hour      sin(2π * hour_of_day / 24)  [−1, 1]
  3  cos_hour      cos(2π * hour_of_day / 24)  [−1, 1]
  4  occupancy     binary flag                 {0, 1}
  5  u_prev        previous normalised action  [0, 1]

Action (1 dim)
--------------
  0  u             normalised HVAC power       [0, 1]
     Physical power: P_hvac = u * P_max  [W]

Reward
------
  r_t = − (λ_c * d_t² + λ_e * P_hvac_t + λ_s * Δu_t²)

  where d_t = comfort deviation as in Parameters.comfort_deviation().
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from smart_hvac.core.parameters import Parameters
from smart_hvac.core.thermal_model import (
    step_rc,
    sinusoidal_temperature,
    load_weather_csv,
    occupancy_schedule,
)


# Normalisation constants (keep observations in roughly [−1, +1])
_T_IN_MIN, _T_IN_MAX = 0.0, 40.0
_T_OUT_MIN, _T_OUT_MAX = -30.0, 50.0


def _norm_temp(T: float, lo: float, hi: float) -> float:
    return (T - lo) / (hi - lo)


class HvacEnv(gym.Env):
    """Single-zone HVAC environment with RC thermal dynamics.

    Parameters
    ----------
    params       : Parameters instance (physical + reward settings)
    seed         : master random seed (reproducible episodes)
    day_type     : 'cold' | 'mild' | 'hot' | 'random'
                   Fixed day mean temperature or sampled each reset.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        params: Parameters | None = None,
        seed: int | None = None,
        day_type: str = "random",
    ) -> None:
        super().__init__()
        self.params = params or Parameters()
        self.day_type = day_type
        self._master_seed = seed
        self.rng = np.random.default_rng(seed)

        # ---- spaces -------------------------------------------------- #
        obs_low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ---- episode state (initialised properly in reset) ------------ #
        self.T_in: float = self.params.T_set
        self.T_mean: float = self.params.T_mean_mild
        self._step: int = 0
        self._u_prev: float = 0.0
        self._occ: np.ndarray = np.zeros(self.params.episode_steps, dtype=np.int8)
        self._weather: np.ndarray | None = None  # preloaded CSV series

        # Preload CSV weather once if needed
        if self.params.weather_profile == "csv":
            self._weather = load_weather_csv(
                self.params.weather_csv_path,  # type: ignore[arg-type]
                self.params.dt,
            )

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Sample day type
        self.T_mean = self._sample_T_mean()

        # Occupancy schedule for this episode
        self._occ = occupancy_schedule(
            self.params.episode_steps,
            self.params.dt,
            self.params.occupancy_profile,
            rng=self.rng,
        )

        # Initial indoor temperature: near comfort band ± 1 °C
        self.T_in = float(
            self.rng.normal(self.params.T_set, 1.0)
        )
        self._step = 0
        self._u_prev = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    #  step                                                                #
    # ------------------------------------------------------------------ #

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        u = float(np.clip(action[0], 0.0, 1.0))
        P_hvac = u * self.params.P_max

        # Current outdoor temperature
        T_out = self._current_T_out()

        # Internal gains (occupancy + noise)
        Q_int = self._current_Q_int()

        # Advance thermal dynamics
        self.T_in = step_rc(
            T_in=self.T_in,
            T_out=T_out,
            P_hvac=P_hvac,
            Q_int=Q_int,
            params=self.params,
        )

        # Reward components
        d = self.params.comfort_deviation(self.T_in)
        r_comfort = -self.params.lambda_c * d ** 2
        r_energy = -self.params.lambda_e * P_hvac
        r_smooth = -self.params.lambda_s * (u - self._u_prev) ** 2
        reward = r_comfort + r_energy + r_smooth

        self._u_prev = u
        self._step += 1

        terminated = False
        truncated = self._step >= self.params.episode_steps

        info: dict[str, float] = {
            "r_comfort": r_comfort,
            "r_energy": r_energy,
            "r_smooth": r_smooth,
            "T_in": self.T_in,
            "T_out": T_out,
            "P_hvac": P_hvac,
            "Q_int": Q_int,
            "comfort_dev": d,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        t_hours = (self._step * self.params.dt) / 3600.0
        omega = 2.0 * math.pi * t_hours / 24.0

        T_out = self._current_T_out()

        obs = np.array(
            [
                _norm_temp(self.T_in, _T_IN_MIN, _T_IN_MAX),
                _norm_temp(T_out, _T_OUT_MIN, _T_OUT_MAX),
                math.sin(omega),
                math.cos(omega),
                float(self._occ[min(self._step, len(self._occ) - 1)]),
                self._u_prev,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _current_T_out(self) -> float:
        if self._weather is not None:
            idx = self._step % len(self._weather)
            return float(self._weather[idx])
        return sinusoidal_temperature(
            step=self._step,
            params=self.params,
            T_mean=self.T_mean,
            rng=self.rng,
        )

    def _current_Q_int(self) -> float:
        occ = float(self._occ[min(self._step, len(self._occ) - 1)])
        Q = occ * self.params.Q_occ
        Q += float(self.rng.normal(0.0, self.params.Q_noise_std))
        return max(0.0, Q)

    def _sample_T_mean(self) -> float:
        mapping = {
            "cold": self.params.T_mean_cold,
            "mild": self.params.T_mean_mild,
            "hot": self.params.T_mean_hot,
        }
        if self.day_type in mapping:
            return mapping[self.day_type]
        # "random": pick one of the three
        return float(self.rng.choice(list(mapping.values())))