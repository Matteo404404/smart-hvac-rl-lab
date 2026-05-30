"""Run a policy in HvacEnv and return trajectory logs."""

from __future__ import annotations
import numpy as np
import pandas as pd

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv


def run_episode(
    policy,
    env: HvacEnv,
    seed: int | None = None,
) -> pd.DataFrame:
    """Roll out one episode and return a DataFrame of per-step info."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    records = []
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step_info["reward"] = reward
        step_info["action"] = float(action[0])
        records.append(step_info)
        done = terminated or truncated

    df = pd.DataFrame(records)
    df["step"] = np.arange(len(df))
    df["t_hours"] = df["step"] * env.params.dt / 3600.0
    return df


def run_n_episodes(
    policy,
    params: Parameters | None = None,
    day_type: str = "random",
    n_episodes: int = 20,
    base_seed: int = 0,
) -> list[pd.DataFrame]:
    """Run n independent episodes and return a list of DataFrames."""
    params = params or Parameters()
    results = []
    for i in range(n_episodes):
        env = HvacEnv(params=params, day_type=day_type, seed=base_seed + i)
        df  = run_episode(policy, env, seed=base_seed + i)
        results.append(df)
    return results
