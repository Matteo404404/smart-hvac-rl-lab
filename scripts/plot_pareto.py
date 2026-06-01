#!/usr/bin/env python
"""Pareto comfort-vs-energy scatter and temperature distribution plots."""
import sys, os
sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from smart_hvac.core.parameters import Parameters
from smart_hvac.control.thermostat import Thermostat
from smart_hvac.control.pid_controller import PIDController
from smart_hvac.evaluation.rollouts import run_n_episodes
from smart_hvac.evaluation.metrics import episode_metrics

COLORS = {"Thermostat": "tab:orange", "PID": "tab:blue",
          "SAC": "tab:green", "PPO": "tab:purple"}

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
    return policies

def main():
    params   = Parameters()
    policies = load_all_policies(params)
    os.makedirs("results/figures", exist_ok=True)

    all_eps = {}
    for name, pol in policies.items():
        all_eps[name] = run_n_episodes(pol, params=params,
                                       day_type="random", n_episodes=20)
        print(f"[{name}] collected 20 episodes")

    # ── Pareto scatter ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, episodes in all_eps.items():
        xs = [(df["P_hvac"] * params.dt / 3_600_000).sum() for df in episodes]
        ys = [df["comfort_dev"].mean() for df in episodes]
        ax.scatter(xs, ys, label=name, alpha=0.7,
                   color=COLORS.get(name), s=60)
        ax.scatter([np.mean(xs)], [np.mean(ys)],
                   marker="D", s=130, color=COLORS.get(name),
                   edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_xlabel("Total Energy Usage [kWh]")
    ax.set_ylabel("Mean Comfort Deviation [K]")
    ax.set_title("All Policies — Comfort vs Energy Trade-off (20 random episodes)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("results/figures/pareto_comfort_energy.png",
                dpi=150, bbox_inches="tight")
    print("Saved -> results/figures/pareto_comfort_energy.png")
    plt.close()

    # ── Temperature distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, episodes in all_eps.items():
        all_T = np.concatenate([df["T_in"].values for df in episodes])
        ax.hist(all_T, bins=40, alpha=0.4, density=True,
                label=name, color=COLORS.get(name))
    ax.axvline(params.T_min, color="green", linestyle="--",
               lw=1.2, label="Comfort band")
    ax.axvline(params.T_max, color="green", linestyle="--", lw=1.2)
    ax.axvspan(params.T_min, params.T_max, alpha=0.08, color="green")
    ax.set_xlabel("Indoor Temperature [C]")
    ax.set_ylabel("Density")
    ax.set_title("All Policies — Indoor Temperature Distribution (20 random episodes)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("results/figures/temperature_distribution.png",
                dpi=150, bbox_inches="tight")
    print("Saved -> results/figures/temperature_distribution.png")
    plt.close()

if __name__ == "__main__":
    main()