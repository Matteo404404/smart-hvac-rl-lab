"""Helpers to load trained SB3 models and expose a unified .act() API."""

from __future__ import annotations
import numpy as np


class SB3Policy:
    """Thin wrapper around a Stable-Baselines3 model."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass  # SB3 models are stateless at inference

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


def load_sac(path: str, **kwargs) -> SB3Policy:
    from stable_baselines3 import SAC
    model = SAC.load(path, **kwargs)
    return SB3Policy(model)


def load_ppo(path: str, **kwargs) -> SB3Policy:
    from stable_baselines3 import PPO
    model = PPO.load(path, **kwargs)
    return SB3Policy(model)
