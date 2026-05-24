"""Helpers to load trained RL models and expose a unified .act() interface."""

from __future__ import annotations
import numpy as np


class LoadedPolicy:
    """Thin wrapper around a loaded SB3 model.

    Exposes act(obs) → float for use with run_rollout(is_baseline=False).
    Also implements .predict() so it's usable directly with SB3 utilities.
    """

    def __init__(self, model_path: str, algo: str = "sac") -> None:
        self.model_path = model_path
        self.algo = algo.lower()
        self._model = self._load()

    def _load(self):
        if self.algo == "sac":
            from stable_baselines3 import SAC
            return SAC.load(self.model_path)
        elif self.algo == "ppo":
            from stable_baselines3 import PPO
            return PPO.load(self.model_path)
        else:
            raise ValueError(f"Unknown algo: {self.algo!r}. Use 'sac' or 'ppo'.")

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        action, state = self._model.predict(obs, deterministic=deterministic)
        return action, state

    def act(self, obs: np.ndarray, deterministic: bool = True) -> float:
        action, _ = self.predict(obs, deterministic=deterministic)
        return float(action[0])