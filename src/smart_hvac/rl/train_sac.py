"""SAC training script.

Usage:
    python -m smart_hvac.rl.train_sac [--timesteps 300000] [--seed 0]
    # or via scripts/train_rl.py

Saves the trained model to results/models/sac_hvac.zip
and evaluation logs to results/logs/sac_eval.csv.
"""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv
from smart_hvac.evaluation.rollouts import run_rollouts
from smart_hvac.evaluation.metrics import aggregate_metrics


# ------------------------------------------------------------------ #
#  Custom logging callback                                             #
# ------------------------------------------------------------------ #

class EpisodeLogCallback(BaseCallback):
    """Logs mean episode reward to a CSV file every eval_freq steps."""

    def __init__(
        self,
        eval_env: HvacEnv,
        log_path: str,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._log_rows: list[dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            trajs = run_rollouts(
                policy=self.model,
                env=self.eval_env,
                n_episodes=self.n_eval_episodes,
                base_seed=999,
                deterministic=True,
                is_baseline=False,
            )
            agg = aggregate_metrics(trajs)
            row = {
                "timestep": self.num_timesteps,
                "mean_reward": agg["mean_reward"]["mean"],
                "std_reward": agg["mean_reward"]["std"],
                "mean_comfort_dev": agg["mean_comfort_dev"]["mean"],
                "pct_in_band": agg["pct_in_band"]["mean"],
                "total_energy_kwh": agg["total_energy_kwh"]["mean"],
            }
            self._log_rows.append(row)
            if self.verbose:
                print(
                    f"  [eval @ {self.num_timesteps:>7d}]"
                    f"  reward={row['mean_reward']:.3f}"
                    f"  comfort_dev={row['mean_comfort_dev']:.3f} K"
                    f"  in_band={row['pct_in_band']:.1f}%"
                )
        return True

    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self._log_rows:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._log_rows)
            print(f"[SAC] Eval log saved → {self.log_path}")


# ------------------------------------------------------------------ #
#  Main training function                                              #
# ------------------------------------------------------------------ #

def train_sac(
    total_timesteps: int = 300_000,
    seed: int = 0,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    buffer_size: int = 200_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    net_arch: list[int] | None = None,
    model_save_path: str = "results/models/sac_hvac",
    log_path: str = "results/logs/sac_eval.csv",
    eval_freq: int = 10_000,
    verbose: int = 1,
) -> SAC:
    """Train a SAC agent on HvacEnv and save the model.

    Returns
    -------
    model : trained SAC model (also saved to model_save_path.zip)
    """
    if net_arch is None:
        net_arch = [64, 64]

    params = Parameters()

    # Vectorised training environments
    def make_env():
        return HvacEnv(params=params, day_type="random")

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)

    # Single deterministic eval environment
    eval_env = HvacEnv(params=params, day_type="mild", seed=42)

    # SAC model
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
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

    print(f"[SAC] Training for {total_timesteps:,} timesteps on {n_envs} parallel envs …")
    model.learn(total_timesteps=total_timesteps, callback=log_callback)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[SAC] Model saved → {model_save_path}.zip")

    vec_env.close()
    return model


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC on HvacEnv")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--model-path", type=str, default="results/models/sac_hvac")
    p.add_argument("--log-path", type=str, default="results/logs/sac_eval.csv")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_sac(
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        model_save_path=args.model_path,
        log_path=args.log_path,
        verbose=args.verbose,
    )