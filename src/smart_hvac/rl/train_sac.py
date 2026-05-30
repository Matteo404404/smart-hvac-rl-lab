"""SAC training script using Stable-Baselines3."""

from __future__ import annotations
import os
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv


def make_env(day_type: str = "random", seed: int = 0):
    def _init():
        env = HvacEnv(params=Parameters(), day_type=day_type, seed=seed)
        return env
    return _init


def train(
    total_timesteps: int = 300_000,
    n_envs: int = 4,
    save_path: str = "results/models/sac_hvac",
    log_path: str = "results/logs/",
    day_type: str = "random",
    seed: int = 42,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Vectorised training envs
    train_env = make_vec_env(
        make_env(day_type=day_type, seed=seed),
        n_envs=n_envs,
    )

    # Separate eval env (fixed seed)
    eval_env = HvacEnv(params=Parameters(), day_type="cold", seed=seed)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path + "_best",
        log_path=log_path,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(save_path)
    print(f"✅  SAC model saved to {save_path}.zip")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs",   type=int, default=4)
    parser.add_argument("--day-type", type=str, default="random")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        day_type=args.day_type,
        seed=args.seed,
    )
