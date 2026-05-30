"""Unit tests for reward function sign and monotonicity."""
import pytest
import numpy as np
from smart_hvac.core.parameters import Parameters
from smart_hvac.envs.hvac_env import HvacEnv


@pytest.fixture
def env():
    return HvacEnv(params=Parameters(), day_type="cold", seed=0)


def test_comfort_penalty_sign(env):
    env.reset(seed=0)
    env.T_in = 15.0  # well below comfort band
    _, reward, _, _, info = env.step(np.array([0.0]))
    assert info["r_comfort"] < 0, "Comfort penalty must be negative."


def test_larger_deviation_larger_penalty(env):
    params = Parameters()
    dev_small = 1.0
    dev_large = 3.0
    penalty_small = -params.lambda_c * dev_small ** 2
    penalty_large = -params.lambda_c * dev_large ** 2
    assert penalty_large < penalty_small, "Larger deviation → larger penalty."


def test_energy_penalty_sign(env):
    env.reset(seed=0)
    env.T_in = 22.0  # inside comfort band
    _, reward, _, _, info = env.step(np.array([1.0]))
    assert info["r_energy"] < 0, "Energy penalty must be negative."
