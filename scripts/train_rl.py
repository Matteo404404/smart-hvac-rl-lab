#!/usr/bin/env python
"""Convenience wrapper: train SAC (and optionally PPO)."""
import sys
sys.path.insert(0, "src")

import argparse
from smart_hvac.rl.train_sac import train as train_sac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",      default="sac", choices=["sac", "ppo"])
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--day-type",  default="random")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    if args.algo == "sac":
        train_sac(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            day_type=args.day_type,
            seed=args.seed,
        )
    else:
        from smart_hvac.rl.train_ppo import train as train_ppo
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            day_type=args.day_type,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
