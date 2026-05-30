"""Aggregate metrics from episode DataFrames."""

from __future__ import annotations
import numpy as np
import pandas as pd


def episode_metrics(df: pd.DataFrame, dt: float = 900.0) -> dict:
    """Compute scalar metrics from one episode trajectory."""
    return {
        "mean_comfort_dev":  float(df["comfort_dev"].mean()),
        "pct_in_band":       float((df["comfort_dev"] == 0.0).mean() * 100),
        "total_energy_kwh":  float((df["P_hvac"] * dt / 3_600_000).sum()),
        "mean_power_w":      float(df["P_hvac"].mean()),
        "total_reward":      float(df["reward"].sum()),
    }


def aggregate_metrics(episodes: list[pd.DataFrame], dt: float = 900.0) -> pd.DataFrame:
    """Aggregate metrics across multiple episodes."""
    rows = [episode_metrics(df, dt) for df in episodes]
    return pd.DataFrame(rows).describe()
