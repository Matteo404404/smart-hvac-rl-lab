"""Run thermostat and PID baselines and print a comparison table.

Usage:
    python scripts/run_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv
from smart_hvac.control.thermostat import Thermostat
from smart_hvac.control.pid_controller import PIDController
from smart_hvac.evaluation.rollouts import run_rollouts
from smart_hvac.evaluation.metrics import aggregate_metrics, metrics_table
from smart_hvac.evaluation.plots import plot_timeseries, plot_pareto

import matplotlib.pyplot as plt
import os

N_EPISODES = 20
SEED = 0

params = Parameters()

policies = {
    "Thermostat": (Thermostat(params=params), True),
    "PID":        (PIDController(params=params), True),
}

all_metrics: dict[str, dict[str, float]] = {}
all_episode_metrics: dict[str, list] = {}
sample_trajs: dict[str, dict] = {}

for name, (policy, is_bl) in policies.items():
    env = HvacEnv(params=params, day_type="cold", seed=SEED)
    trajs = run_rollouts(
        policy=policy,
        env=env,
        n_episodes=N_EPISODES,
        base_seed=SEED,
        is_baseline=is_bl,
    )
    from smart_hvac.evaluation.metrics import compute_metrics
    ep_metrics = [compute_metrics(t) for t in trajs]
    agg = aggregate_metrics(trajs)
    all_metrics[name] = {k: v["mean"] for k, v in agg.items()}
    all_episode_metrics[name] = ep_metrics
    sample_trajs[name] = trajs[0]   # first episode for time-series

print("\n=== Baseline Comparison (cold day, 20 episodes) ===\n")
print(metrics_table(all_metrics))

# Save figures
os.makedirs("results/figures", exist_ok=True)

fig1 = plot_timeseries(sample_trajs, T_min=params.T_min, T_max=params.T_max)
fig1.savefig("results/figures/baselines_timeseries.png", dpi=150)

fig2 = plot_pareto(all_episode_metrics)
fig2.savefig("results/figures/baselines_pareto.png", dpi=150)

print("\nFigures saved to results/figures/")
plt.close("all")