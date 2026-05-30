"""Gymnasium-compatible HvacEnv.

Observation vector (6-dim):
    [T_in, T_out, sin(ω·t), cos(ω·t), occupancy, u_prev]

Action space:
    Box([0], [1]) — normalised HVAC power fraction
"""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from smart_hvac.core.parameters import Parameters
from smart_hvac.core.thermal_model import SinusoidalWeather, occupancy_flag, step_rc


class HvacEnv(gym.Env):
    """Single-zone HVAC control environment with RC thermal dynamics."""

    metadata = {"render_modes": ["human"]}

    OBS_DIM = 6
    ACT_DIM = 1

    def __init__(
        self,
        params: Parameters | None = None,
        day_type: str = "random",
        seed: int | None = None,
    ):
        super().__init__()
        self.params = params or Parameters()
        self.day_type = day_type          # "cold" | "mild" | "hot" | "random"

        # Gymnasium spaces
        obs_low  = np.array([0.0, -30.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([40.0, 50.0,  1.0,  1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Internal state (initialised in reset)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._weather = SinusoidalWeather(self.params, self._rng)
        self.T_in: float = 22.0
        self.T_out: float = 10.0
        self.t_step: int = 0
        self.u_prev: float = 0.0
        self._current_day_type: str = "mild"

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._weather = SinusoidalWeather(self.params, self._rng)

        # Choose day type
        if self.day_type == "random":
            self._current_day_type = self._rng.choice(["cold", "mild", "hot"])
        else:
            self._current_day_type = self.day_type

        self._weather.reset(self._current_day_type)
        self.T_in  = float(self._rng.normal(22.0, 0.5))
        self.T_out = self._weather.get(0)
        self.t_step = 0
        self.u_prev = 0.0
        return self._get_obs(), {"day_type": self._current_day_type}

    def step(self, action: np.ndarray):
        u = float(np.clip(action[0], 0.0, 1.0))
        P_hvac = u * self.params.P_max

        # Internal gains
        occ = occupancy_flag(self.t_step, self.params.occupancy_profile, self.params.dt)
        Q_int = occ * self.params.Q_occ + float(
            self._rng.normal(0.0, self.params.Q_noise_std)
        )

        # Thermal update
        self.T_out = self._weather.get(self.t_step)
        self.T_in  = step_rc(self.T_in, self.T_out, P_hvac, Q_int, self.params)

        # Reward
        d = self._comfort_deviation(self.T_in)
        r_comfort = -self.params.lambda_c * d ** 2
        r_energy  = -self.params.lambda_e * P_hvac
        r_smooth  = -self.params.lambda_s * (u - self.u_prev) ** 2
        r_band    = self.params.lambda_b * float(d == 0.0)
        reward    = r_comfort + r_energy + r_smooth + r_band

        self.u_prev = u
        self.t_step += 1
        truncated = self.t_step >= self.params.episode_steps

        info = dict(
            r_band=r_band,
            r_comfort=r_comfort,
            r_energy=r_energy,
            r_smooth=r_smooth,
            T_in=self.T_in,
            T_out=self.T_out,
            P_hvac=P_hvac,
            Q_int=Q_int,
            occupancy=occ,
            comfort_dev=d,
        )
        return self._get_obs(), reward, False, truncated, info

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        t_hours = self.t_step * self.params.dt / 3600.0
        omega   = 2 * math.pi / 24.0
        occ = float(occupancy_flag(self.t_step, self.params.occupancy_profile, self.params.dt))
        return np.array([
            self.T_in,
            self.T_out,
            math.sin(omega * t_hours),
            math.cos(omega * t_hours),
            occ,
            self.u_prev,
        ], dtype=np.float32)

    def _comfort_deviation(self, T: float) -> float:
        if T < self.params.T_min:
            return self.params.T_min - T
        elif T > self.params.T_max:
            return T - self.params.T_max
        return 0.0
