"""Plotting utilities for trajectory analysis.

All functions return a matplotlib Figure so the caller can either
show() or savefig() as needed.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ------------------------------------------------------------------ #
#  Style defaults                                                      #
# ------------------------------------------------------------------ #

POLICY_COLORS = {
    "Thermostat": "#e07b39",
    "PID": "#5b8db8",
    "SAC": "#3aaa5e",
    "PPO": "#a66fc4",
}

def _policy_color(name: str) -> str:
    for k, v in POLICY_COLORS.items():
        if k.lower() in name.lower():
            return v
    return "#888888"


# ------------------------------------------------------------------ #
#  1. Time-series: temperature + power                                 #
# ------------------------------------------------------------------ #

def plot_timeseries(
    trajs: dict[str, dict[str, np.ndarray]],
    params_dt_min: float = 15.0,
    T_min: float = 21.0,
    T_max: float = 23.0,
    title: str = "Temperature and HVAC Power – Single Episode",
) -> plt.Figure:
    """Plot T_in, T_out and P_hvac time-series for multiple policies.

    Parameters
    ----------
    trajs         : dict  policy_name → trajectory dict
    params_dt_min : time-step duration in minutes
    T_min / T_max : comfort band boundaries [°C]
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    time_h = (
        np.arange(len(next(iter(trajs.values()))["T_in"])) * params_dt_min / 60.0
    )

    first = True
    for name, traj in trajs.items():
        color = _policy_color(name)
        ax1.plot(time_h, traj["T_in"], label=name, color=color, linewidth=1.8)
        ax2.plot(
            time_h,
            traj["P_hvac"] / 1000.0,
            label=name,
            color=color,
            linewidth=1.8,
        )
        if first:
            ax1.plot(
                time_h, traj["T_out"],
                label="T_out", color="#aaaaaa",
                linewidth=1.2, linestyle="--",
            )
            first = False

    # Comfort band shading
    ax1.axhspan(T_min, T_max, alpha=0.10, color="#3aaa5e", label="Comfort band")
    ax1.axhline(T_min, color="#3aaa5e", linewidth=0.8, linestyle=":")
    ax1.axhline(T_max, color="#3aaa5e", linewidth=0.8, linestyle=":")

    ax1.set_ylabel("Temperature [°C]")
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("HVAC Power [kW]")
    ax2.set_xlabel("Time of day [h]")
    ax2.legend(fontsize=9, ncol=3)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  2. Indoor temperature histogram                                     #
# ------------------------------------------------------------------ #

def plot_temperature_distribution(
    trajs: dict[str, dict[str, np.ndarray]],
    T_min: float = 21.0,
    T_max: float = 23.0,
    bins: int = 40,
    title: str = "Indoor Temperature Distribution",
) -> plt.Figure:
    """Plot overlapping histograms of T_in for multiple policies."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, traj in trajs.items():
        ax.hist(
            traj["T_in"],
            bins=bins,
            alpha=0.45,
            label=name,
            color=_policy_color(name),
            density=True,
        )

    ax.axvspan(T_min, T_max, alpha=0.10, color="#3aaa5e")
    ax.axvline(T_min, color="#3aaa5e", linewidth=1.0, linestyle="--")
    ax.axvline(T_max, color="#3aaa5e", linewidth=1.0, linestyle="--", label="Comfort band")

    ax.set_xlabel("Indoor Temperature [°C]")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  3. Comfort–energy Pareto scatter                                    #
# ------------------------------------------------------------------ #

def plot_pareto(
    policy_episode_metrics: dict[str, list[dict[str, float]]],
    title: str = "Comfort vs Energy Trade-off",
) -> plt.Figure:
    """Scatter plot: comfort deviation (y) vs energy usage (x) per episode.

    Parameters
    ----------
    policy_episode_metrics : dict  policy_name → list of per-episode metrics dicts
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, ep_metrics in policy_episode_metrics.items():
        x = [m["total_energy_kwh"] for m in ep_metrics]
        y = [m["mean_comfort_dev"] for m in ep_metrics]
        color = _policy_color(name)
        ax.scatter(x, y, label=name, color=color, alpha=0.65, s=50, edgecolors="white", linewidths=0.4)
        # Mean marker
        ax.scatter(
            np.mean(x), np.mean(y),
            marker="D", s=120, color=color,
            edgecolors="black", linewidths=0.8, zorder=5,
        )

    ax.set_xlabel("Total Energy Usage [kWh]")
    ax.set_ylabel("Mean Comfort Deviation [K]")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  4. Cumulative cost curves (causal day experiment)                   #
# ------------------------------------------------------------------ #

def plot_cumulative_costs(
    trajs: dict[str, dict[str, np.ndarray]],
    params_dt_min: float = 15.0,
    lambda_c: float = 1.0,
    lambda_e: float = 1e-4,
    title: str = "Cumulative Cost – Fixed Day Causal Experiment",
) -> plt.Figure:
    """Plot cumulative comfort cost and energy cost over a fixed episode."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    time_h = (
        np.arange(len(next(iter(trajs.values()))["comfort_dev"])) * params_dt_min / 60.0
    )

    for name, traj in trajs.items():
        color = _policy_color(name)
        cum_comfort = np.cumsum(lambda_c * traj["comfort_dev"] ** 2)
        cum_energy = np.cumsum(lambda_e * traj["P_hvac"])
        ax1.plot(time_h, cum_comfort, label=name, color=color, linewidth=1.8)
        ax2.plot(time_h, cum_energy, label=name, color=color, linewidth=1.8)

    ax1.set_title("Cumulative Comfort Cost")
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("Σ λ_c · d²")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Cumulative Energy Cost")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Σ λ_e · P_hvac")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  5. Training curve                                                   #
# ------------------------------------------------------------------ #

def plot_training_curve(
    timesteps: np.ndarray,
    mean_rewards: np.ndarray,
    std_rewards: np.ndarray | None = None,
    label: str = "SAC",
    title: str = "RL Training Curve",
) -> plt.Figure:
    """Plot mean episode reward vs training timesteps with optional std band."""
    fig, ax = plt.subplots(figsize=(10, 5))
    color = _policy_color(label)
    ax.plot(timesteps, mean_rewards, label=label, color=color, linewidth=2.0)

    if std_rewards is not None:
        ax.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.20,
            color=color,
        )

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig