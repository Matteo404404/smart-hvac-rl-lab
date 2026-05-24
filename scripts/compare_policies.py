"""Compare all policies (thermostat, PID, SAC, PPO) on a fixed causal day.

Usage:
    python scripts/compare_policies.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv
from smart_hvac.control.thermostat import Thermostat
from smart_hvac.control.pid_controller import PIDController
from smart_hvac.rl.policies import LoadedPolicy
from smart_hvac.evaluation.rollouts import run_rollout
from smart_hvac.evaluation.metrics import compute_metrics, metrics_table
from smart_hvac.evaluation.plots import (
    plot_timeseries,
    plot_temperature_distribution,
    plot_cumulative_costs,
)

CAUSAL_SEED = 42   # same seed → identical weather + occupancy for all
params = Parameters()

policies = {}

# Baselines
policies["Thermostat"] = (Thermostat(params=params), True)
policies["PID"]        = (PIDController(params=params), True)

# RL models (skip gracefully if not trained yet)
for algo in ("sac", "ppo"):
    path = f"results/models/{algo}_hvac"
    if os.path.exists(path + ".zip"):
        policies[algo.upper()] = (LoadedPolicy(path, algo=algo), False)
    else:
        print(f"[skip] {algo.upper()} model not found at {path}.zip")

# Run each policy on the exact same causal day
sample_trajs: dict[str, dict] = {}
episode_metrics: dict[str, dict] = {}

for name, (policy, is_bl) in policies.items():
    env = HvacEnv(params=params, day_type="cold", seed=CAUSAL_SEED)
    traj = run_rollout(
        policy=policy,
        env=env,
        seed=CAUSAL_SEED,
        deterministic=True,
        is_baseline=is_bl,
    )
    sample_trajs[name] = traj
    episode_metrics[name] = compute_metrics(traj)

print("\n=== Causal Day Comparison (seed=42, cold day) ===\n")
print(metrics_table(episode_metrics))

os.makedirs("results/figures", exist_ok=True)

fig1 = plot_timeseries(sample_trajs, T_min=params.T_min, T_max=params.T_max,
                       title="Causal Day – Temperature & Power")
fig1.savefig("results/figures/causal_timeseries.png", dpi=150)

fig2 = plot_temperature_distribution(sample_trajs, T_min=params.T_min, T_max=params.T_max,
                                     title="Causal Day – Temperature Distribution")
fig2.savefig("results/figures/causal_temp_dist.png", dpi=150)

fig3 = plot_cumulative_costs(
    sample_trajs,
    lambda_c=params.lambda_c,
    lambda_e=params.lambda_e,
    title="Causal Day – Cumulative Costs",
)
fig3.savefig("results/figures/causal_cumulative_costs.png", dpi=150)

print("\nFigures saved to results/figures/")
plt.close("all")