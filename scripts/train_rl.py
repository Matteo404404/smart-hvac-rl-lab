"""Convenience wrapper to train SAC (and optionally PPO) from CLI.

Usage:
    python scripts/train_rl.py
    python scripts/train_rl.py --algo ppo --timesteps 500000
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    if args.algo == "sac":
        from smart_hvac.rl.train_sac import train_sac
        train_sac(
            total_timesteps=args.timesteps,
            seed=args.seed,
            n_envs=args.n_envs,
            verbose=args.verbose,
        )
    else:
        from smart_hvac.rl.train_ppo import train_ppo
        train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed,
            n_envs=args.n_envs,
            verbose=args.verbose,
        )

if __name__ == "__main__":
    main()