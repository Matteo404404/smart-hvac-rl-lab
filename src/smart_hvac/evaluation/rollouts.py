"""Episode rollout utilities.

Runs a policy (RL agent or baseline controller) for one or more episodes
and collects full trajectories for later analysis.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol
import numpy as np

from smart_hvac.envs.hvac_env import HvacEnv
from smart_hvac.core.parameters import Parameters


# ------------------------------------------------------------------ #
#  Policy protocol – any object with .act(obs) or .predict(obs)       #
# ------------------------------------------------------------------ #

class BaselinePolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, T_in: float) -> float: ...


class RLPolicy(Protocol):
    def predict(
        self, obs: np.ndarray, deterministic: bool
    ) -> tuple[np.ndarray, Any]: ...


# ------------------------------------------------------------------ #
#  Single rollout                                                      #
# ------------------------------------------------------------------ #

def run_rollout(
    policy,
    env: HvacEnv,
    seed: int | None = None,
    deterministic: bool = True,
    is_baseline: bool = False,
) -> dict[str, np.ndarray]:
    """Run one episode and return a trajectory dict.

    Parameters
    ----------
    policy       : RL policy (SB3 model) or baseline (Thermostat/PID)
    env          : HvacEnv instance
    seed         : episode seed for reproducibility
    deterministic: use deterministic actions (RL only)
    is_baseline  : if True, call policy.reset() then policy.act(T_in)
                   instead of policy.predict(obs)

    Returns
    -------
    traj : dict with arrays of length episode_steps:
        T_in, T_out, P_hvac, Q_int, action, reward,
        r_comfort, r_energy, r_smooth, comfort_dev, occupancy
    """
    obs, _ = env.reset(seed=seed)
    if is_baseline:
        policy.reset()

    T_ins, T_outs, P_hvacs = [], [], []
    Q_ints, actions, rewards = [], [], []
    r_comforts, r_energies, r_smooths = [], [], []
    comfort_devs, occupancies = [], []

    done = False
    while not done:
        if is_baseline:
            u = policy.act(env.T_in)
            action = np.array([u], dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        T_ins.append(info["T_in"])
        T_outs.append(info["T_out"])
        P_hvacs.append(info["P_hvac"])
        Q_ints.append(info["Q_int"])
        actions.append(float(action[0]))
        rewards.append(float(reward))
        r_comforts.append(info["r_comfort"])
        r_energies.append(info["r_energy"])
        r_smooths.append(info["r_smooth"])
        comfort_devs.append(info["comfort_dev"])
        occupancies.append(
            float(env._occ[min(env._step - 1, len(env._occ) - 1)])
        )

    return {
        "T_in": np.array(T_ins, dtype=np.float32),
        "T_out": np.array(T_outs, dtype=np.float32),
        "P_hvac": np.array(P_hvacs, dtype=np.float32),
        "Q_int": np.array(Q_ints, dtype=np.float32),
        "action": np.array(actions, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "r_comfort": np.array(r_comforts, dtype=np.float32),
        "r_energy": np.array(r_energies, dtype=np.float32),
        "r_smooth": np.array(r_smooths, dtype=np.float32),
        "comfort_dev": np.array(comfort_devs, dtype=np.float32),
        "occupancy": np.array(occupancies, dtype=np.float32),
    }


# ------------------------------------------------------------------ #
#  Multi-seed rollouts                                                 #
# ------------------------------------------------------------------ #

def run_rollouts(
    policy,
    env: HvacEnv,
    n_episodes: int = 10,
    base_seed: int = 0,
    deterministic: bool = True,
    is_baseline: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Run multiple episodes and return a list of trajectory dicts.

    Parameters
    ----------
    n_episodes : number of episodes to evaluate
    base_seed  : seeds will be base_seed, base_seed+1, …

    Returns
    -------
    trajs : list of trajectory dicts (one per episode)
    """
    return [
        run_rollout(
            policy=policy,
            env=env,
            seed=base_seed + i,
            deterministic=deterministic,
            is_baseline=is_baseline,
        )
        for i in range(n_episodes)
    ]