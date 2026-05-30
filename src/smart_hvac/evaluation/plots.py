"""Plotting utilities for policy evaluation."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from smart_hvac.core.parameters import Parameters


COLORS = {"Thermostat": "tab:orange", "PID": "tab:blue", "SAC": "tab:green", "PPO": "tab:purple"}


def plot_temperature_trajectories(
    trajectories: dict[str, pd.DataFrame],
    params: Parameters | None = None,
    title: str = "Indoor Temperature Trajectories",
    ax=None,
    figsize=(12, 4),
):
    params = params or Parameters()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for name, df in trajectories.items():
        ax.plot(df["t_hours"], df["T_in"], label=name, color=COLORS.get(name))
    ax.axhspan(params.T_min, params.T_max, alpha=0.1, color="green", label="Comfort band")
    ax.axhline(params.T_min, color="green", linestyle="--", linewidth=0.8)
    ax.axhline(params.T_max, color="green", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time of day [h]")
    ax.set_ylabel("T_in [°C]")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_power_trajectories(
    trajectories: dict[str, pd.DataFrame],
    title: str = "HVAC Power Trajectories",
    ax=None,
    figsize=(12, 3),
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for name, df in trajectories.items():
        ax.plot(df["t_hours"], df["P_hvac"] / 1000, label=name, color=COLORS.get(name))
    ax.set_xlabel("Time of day [h]")
    ax.set_ylabel("P_hvac [kW]")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_comfort_energy_pareto(
    policy_episodes: dict[str, list[pd.DataFrame]],
    params: Parameters | None = None,
    figsize=(10, 6),
    save_path: str | None = None,
):
    params = params or Parameters()
    fig, ax = plt.subplots(figsize=figsize)
    dt = params.dt
    for name, episodes in policy_episodes.items():
        xs, ys = [], []
        for df in episodes:
            xs.append((df["P_hvac"] * dt / 3_600_000).sum())
            ys.append(df["comfort_dev"].mean())
        ax.scatter(xs, ys, label=name, alpha=0.7, color=COLORS.get(name))
    ax.set_xlabel("Total Energy Usage [kWh]")
    ax.set_ylabel("Mean Comfort Deviation [K]")
    ax.set_title("All Policies — Comfort vs Energy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
