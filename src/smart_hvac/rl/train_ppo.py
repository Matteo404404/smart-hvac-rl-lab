"""PPO training script (alternative to SAC).

Usage:
    python -m smart_hvac.rl.train_ppo [--timesteps 500000] [--seed 0]
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv
from smart_hvac.rl.train_sac import EpisodeLogCallback


def train_ppo(
    total_timesteps: int = 500_000,
    seed: int = 0,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    net_arch: list[int] | None = None,
    model_save_path: str = "results/models/ppo_hvac",
    log_path: str = "results/logs/ppo_eval.csv",
    eval_freq: int = 10_000,
    verbose: int = 1,
) -> PPO:
    """Train a PPO agent on HvacEnv and save the model."""
    if net_arch is None:
        net_arch = [64, 64]

    params = Parameters()

    def make_env():
        return HvacEnv(params=params, day_type="random")

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    eval_env = HvacEnv(params=params, day_type="mild", seed=42)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": net_arch},
        verbose=verbose,
        seed=seed,
    )

    log_callback = EpisodeLogCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        verbose=verbose,
    )

    print(f"[PPO] Training for {total_timesteps:,} timesteps on {n_envs} parallel envs …")
    model.learn(total_timesteps=total_timesteps, callback=log_callback)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[PPO] Model saved → {model_save_path}.zip")

    vec_env.close()
    return model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on HvacEnv")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--model-path", type=str, default="results/models/ppo_hvac")
    p.add_argument("--log-path", type=str, default="results/logs/ppo_eval.csv")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        model_save_path=args.model_path,
        log_path=args.log_path,
        verbose=args.verbose,
    )