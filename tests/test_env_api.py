"""Unit tests for Gymnasium API compliance of HvacEnv."""

import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env

from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv


@pytest.fixture
def env():
    return HvacEnv(params=Parameters(), seed=0)


def test_reset_returns_correct_shapes(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_step_returns_correct_types(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_obs_within_bounds(env):
    obs, _ = env.reset(seed=0)
    for _ in range(env.params.episode_steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.all(obs >= env.observation_space.low - 1e-5)
        assert np.all(obs <= env.observation_space.high + 1e-5)
        if terminated or truncated:
            break


def test_episode_terminates_after_correct_steps(env):
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        steps += 1
        done = terminated or truncated
    assert steps == env.params.episode_steps


def test_gymnasium_check(env):
    """Run the official Gymnasium env checker."""
    check_env(env, warn=True)