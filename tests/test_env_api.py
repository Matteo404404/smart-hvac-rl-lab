"""Unit tests for Gymnasium API compliance."""
import numpy as np
import pytest
from smart_hvac.envs.hvac_env import HvacEnv
from smart_hvac.core.parameters import Parameters


@pytest.fixture
def env():
    return HvacEnv(params=Parameters(), day_type="mild", seed=42)


def test_reset_returns_correct_shape(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (6,)
    assert "day_type" in info


def test_step_returns_correct_types(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates(env):
    env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps == env.params.episode_steps


def test_obs_in_bounds(env):
    env.reset(seed=0)
    for _ in range(env.params.episode_steps):
        action = env.action_space.sample()
        obs, _, _, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs), f"obs {obs} out of bounds"
        if truncated:
            break
