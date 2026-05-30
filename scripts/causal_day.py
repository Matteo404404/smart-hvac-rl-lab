#!/usr/bin/env python
"""Causal day experiment: same seed, all policies, fixed exogenous conditions."""
import sys, os
sys.path.insert(0, "src")
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from smart_hvac.core.parameters import Parameters
from smart_hvac.control.thermostat import Thermostat
from smart_hvac.control.pid_controller import PIDController
from smart_hvac.evaluation.rollouts import run_episode
from smart_hvac.envs.hvac_env import HvacEnv

COLORS = {"Thermostat": "tab:orange", "PID": "tab:blue",
          "SAC": "tab:green", "PPO": "tab:purple"}

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
    return policies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--day-type", default="cold")
    parser.add_argument("--out-dir",  default="results/figures")
    args = parser.parse_args()
    params   = Parameters()
    policies = load_policies(params)
    os.makedirs(args.out_dir, exist_ok=True)

    trajs = {}
    for name, pol in policies.items():
        env = HvacEnv(params=params, day_type=args.day_type, seed=args.seed)
        df  = run_episode(pol, env, seed=args.seed)
        trajs[name] = df
        print(f"[{name}] comfort_dev={df['comfort_dev'].mean():.3f}  "
              f"pct_in_band={100*(df['comfort_dev']==0).mean():.1f}%  "
              f"energy={(df['P_hvac']*params.dt/3_600_000).sum():.2f} kWh")

    # Plot 1: temperature + power
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7), sharex=True)
    for name, df in trajs.items():
        ax1.plot(df["t_hours"], df["T_in"], label=name, color=COLORS.get(name))
    if "T_out" in list(trajs.values())[0].columns:
        df0 = list(trajs.values())[0]
        ax1.plot(df0["t_hours"], df0["T_out"], "k--", alpha=0.4, label="T_out")
    ax1.axhspan(params.T_min, params.T_max, alpha=0.1, color="green", label="Comfort band")
    ax1.axhline(params.T_min, color="green", linestyle="--", lw=0.8)
    ax1.axhline(params.T_max, color="green", linestyle="--", lw=0.8)
    ax1.set_ylabel("Temperature [C]")
    ax1.set_title(f"Causal Day - All Policies, Identical Exogenous (seed={args.seed})")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    for name, df in trajs.items():
        ax2.plot(df["t_hours"], df["P_hvac"]/1000, label=name, color=COLORS.get(name))
    ax2.set_ylabel("HVAC Power [kW]"); ax2.set_xlabel("Time [h]")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(args.out_dir, f"causal_day_{args.day_type}_seed{args.seed}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p}"); plt.close()

    # Plot 2: cumulative costs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
    for name, df in trajs.items():
        c = COLORS.get(name)
        ax1.plot(df["t_hours"], (params.lambda_c * df["comfort_dev"]**2).cumsum(),
                 label=name, color=c)
        ax2.plot(df["t_hours"], (params.lambda_e * df["P_hvac"]).cumsum(),
                 label=name, color=c)
    for ax, title, ylabel in [
        (ax1, "Cumulative Comfort Cost", "sum(lc * d^2)"),
        (ax2, "Cumulative Energy Cost",  "sum(le * P_hvac)")]:
        ax.set_title(title); ax.set_xlabel("Time [h]"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Causal Day - Cumulative Comfort & Energy Cost")
    plt.tight_layout()
    p2 = os.path.join(args.out_dir, f"causal_cumcost_{args.day_type}_seed{args.seed}.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"Saved -> {p2}"); plt.close()

    # Summary table
    rows = []
    for name, df in trajs.items():
        rows.append({
            "Policy":           name,
            "Comfort Dev [K]":  round(df["comfort_dev"].mean(), 3),
            "% In Band":        round(100*(df["comfort_dev"]==0).mean(), 1),
            "Energy [kWh]":     round((df["P_hvac"]*params.dt/3_600_000).sum(), 3),
            "Total Reward":     round(df["reward"].sum(), 1),
        })
    summary = pd.DataFrame(rows).set_index("Policy")
    print("\n=== Causal Day Summary ===")
    print(summary.to_string())
    summary.to_csv(os.path.join(args.out_dir, f"causal_summary_{args.day_type}.csv"))

if __name__ == "__main__":
    main()