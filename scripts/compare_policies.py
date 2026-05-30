#!/usr/bin/env python
"""Compare all available policies across day types."""
import sys, os
sys.path.insert(0, "src")
import argparse
import pandas as pd
from smart_hvac.core.parameters import Parameters
from smart_hvac.control.thermostat import Thermostat
from smart_hvac.control.pid_controller import PIDController
from smart_hvac.evaluation.rollouts import run_n_episodes
from smart_hvac.evaluation.metrics import episode_metrics

def load_policies(params):
    policies = {"Thermostat": Thermostat(params), "PID": PIDController(params)}
    for algo, path in [("SAC", "results/models/sac_hvac.zip"),
                       ("PPO", "results/models/ppo_hvac.zip")]:
        if os.path.exists(path):
            if algo == "SAC":
                from smart_hvac.rl.policies import load_sac
                policies[algo] = load_sac(path)
            else:
                from smart_hvac.rl.policies import load_ppo
                policies[algo] = load_ppo(path)
        else:
            print(f"[skip] {algo} not found at {path}")
    return policies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-type",   default="cold")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--save-csv",   default="results/logs/comparison.csv")
    args = parser.parse_args()
    params   = Parameters()
    policies = load_policies(params)
    rows = []
    for name, pol in policies.items():
        eps = run_n_episodes(pol, params=params,
                             day_type=args.day_type,
                             n_episodes=args.n_episodes)
        m = pd.DataFrame([episode_metrics(df, params.dt) for df in eps]).mean()
        m["policy"] = name
        rows.append(m)
        print(f"[{name}] done")
    df = pd.DataFrame(rows).set_index("policy")
    cols = ["mean_comfort_dev", "pct_in_band",
            "total_energy_kwh", "mean_power_w", "total_reward"]
    print(f"\n=== {args.day_type} | {args.n_episodes} episodes ===")
    print(df[cols].to_string())
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    df[cols].to_csv(args.save_csv)
    print(f"\nSaved -> {args.save_csv}")

if __name__ == "__main__":
    main()