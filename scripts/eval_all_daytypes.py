#!/usr/bin/env python
"""Evaluate all policies on cold / mild / hot / random day types."""
import sys, os
sys.path.insert(0, "src")
import pandas as pd
from smart_hvac.core.parameters import Parameters
from smart_hvac.control.thermostat import Thermostat
from smart_hvac.control.pid_controller import PIDController
from smart_hvac.evaluation.rollouts import run_n_episodes
from smart_hvac.evaluation.metrics import episode_metrics

def load_all_policies(params):
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
            print(f"[skip] {algo} not found")
    return policies

def main():
    params   = Parameters()
    policies = load_all_policies(params)
    os.makedirs("results/logs", exist_ok=True)
    all_rows = []

    for day_type in ["cold", "mild", "hot", "random"]:
        print(f"\n{'='*55}")
        print(f"  Day type: {day_type}")
        print(f"{'='*55}")
        for name, pol in policies.items():
            eps = run_n_episodes(pol, params=params,
                                 day_type=day_type, n_episodes=20)
            m = pd.DataFrame([episode_metrics(df, params.dt) for df in eps]).mean()
            m["policy"]   = name
            m["day_type"] = day_type
            all_rows.append(m)
            print(f"  [{name:12s}] comfort={m['mean_comfort_dev']:.3f}K  "
                  f"band={m['pct_in_band']:.1f}%  "
                  f"energy={m['total_energy_kwh']:.1f}kWh  "
                  f"reward={m['total_reward']:.1f}")

    df = pd.DataFrame(all_rows).set_index(["day_type", "policy"])
    df.to_csv("results/logs/eval_all_daytypes.csv")
    print("\nSaved -> results/logs/eval_all_daytypes.csv")

if __name__ == "__main__":
    main()