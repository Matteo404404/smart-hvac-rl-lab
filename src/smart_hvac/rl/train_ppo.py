"""PPO training script using Stable-Baselines3."""

from __future__ import annotations
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv


def make_env(day_type: str = "random", seed: int = 0):
    def _init():
        return HvacEnv(params=Parameters(), day_type=day_type, seed=seed)
    return _init


def train(
    total_timesteps: int = 300_000,
    n_envs: int = 4,
    save_path: str = "results/models/ppo_hvac",
    day_type: str = "random",
    seed: int = 42,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = make_vec_env(make_env(day_type=day_type, seed=seed), n_envs=n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅  PPO model saved to {save_path}.zip")
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
