"""Scalar metrics computed from trajectory dicts.

All functions accept a single trajectory dict as returned by
rollouts.run_rollout(), or a list of trajectories for aggregated stats.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def compute_metrics(traj: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute per-episode scalar metrics from a trajectory dict.

    Returns
    -------
    metrics dict with keys:
        mean_comfort_dev   – mean absolute comfort deviation [K]
        max_comfort_dev    – worst-case comfort deviation [K]
        pct_in_band        – % of steps where comfort_dev == 0
        total_energy_kwh   – total HVAC energy used [kWh]
        mean_power_w       – mean HVAC power [W]
        total_reward       – sum of per-step rewards
        mean_reward        – mean per-step reward
        total_r_comfort    – summed comfort reward term
        total_r_energy     – summed energy reward term
        total_r_smooth     – summed smoothness reward term
    """
    d = traj["comfort_dev"]
    P = traj["P_hvac"]
    r = traj["reward"]
    n = len(r)

    # Infer dt from P and energy:  we need dt to compute kWh
    # dt is not stored in traj; derive from the env or use a default.
    # Since we don't have params here, we compute in [W·steps] and
    # let the caller scale if needed. For convenience we assume 15-min steps.
    dt_h = 15.0 / 60.0  # hours per step (15 min default)

    return {
        "mean_comfort_dev": float(np.mean(d)),
        "max_comfort_dev": float(np.max(d)),
        "pct_in_band": float(np.mean(d == 0.0) * 100.0),
        "total_energy_kwh": float(np.sum(P) * dt_h / 1000.0),
        "mean_power_w": float(np.mean(P)),
        "total_reward": float(np.sum(r)),
        "mean_reward": float(np.mean(r)),
        "total_r_comfort": float(np.sum(traj["r_comfort"])),
        "total_r_energy": float(np.sum(traj["r_energy"])),
        "total_r_smooth": float(np.sum(traj["r_smooth"])),
    }


def aggregate_metrics(
    trajs: list[dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics over multiple episodes.

    Returns
    -------
    dict mapping metric_name → {"mean": …, "std": …, "min": …, "max": …}
    """
    all_metrics = [compute_metrics(t) for t in trajs]
    keys = list(all_metrics[0].keys())
    result: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        result[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return result


def metrics_table(
    policy_metrics: dict[str, dict[str, float]],
) -> str:
    """Pretty-print a comparison table.

    Parameters
    ----------
    policy_metrics : dict mapping policy_name → metrics dict
                     (as returned by compute_metrics or the 'mean'
                      slice of aggregate_metrics)

    Returns
    -------
    Formatted string table for printing or logging.
    """
    keys = [
        "mean_comfort_dev",
        "pct_in_band",
        "total_energy_kwh",
        "mean_power_w",
        "total_reward",
    ]
    col_w = 18
    header = f"{'Metric':<25}" + "".join(f"{name:>{col_w}}" for name in policy_metrics)
    sep = "-" * len(header)
    rows = [header, sep]
    for k in keys:
        row = f"{k:<25}"
        for m in policy_metrics.values():
            row += f"{m.get(k, float('nan')):>{col_w}.4f}"
        rows.append(row)
    return "\n".join(rows)