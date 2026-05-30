#!/usr/bin/env python
"""Quick CLI to evaluate baselines on cold/mild/hot days."""
import sys
sys.path.insert(0, "src")

import argparse
import pandas as pd
from smart_hvac.core.parameters import Parameters
from smart_hvac.control.thermostat import Thermostat
from smart_hvac.control.pid_controller import PIDController
from smart_hvac.evaluation.rollouts import run_n_episodes
from smart_hvac.evaluation.metrics import episode_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-type", default="cold")
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    params = Parameters()
    policies = {"Thermostat": Thermostat(params), "PID": PIDController(params)}

    for name, policy in policies.items():
        episodes = run_n_episodes(
            policy, params=params, day_type=args.day_type, n_episodes=args.n_episodes
        )
        rows = [episode_metrics(df, params.dt) for df in episodes]
        df_agg = pd.DataFrame(rows).mean()
        print(f"\n=== {name} | {args.day_type} | {args.n_episodes} episodes ===")
        print(df_agg.to_string())


if __name__ == "__main__":
    main()
