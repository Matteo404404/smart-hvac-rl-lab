"""Unit tests for reward components."""

import pytest
from smart_hvac.core.parameters import Parameters


@pytest.fixture
def params():
    return Parameters()


def test_comfort_deviation_inside_band(params):
    assert params.comfort_deviation(22.0) == 0.0


def test_comfort_deviation_below_band(params):
    d = params.comfort_deviation(20.0)
    assert d == pytest.approx(1.0)


def test_comfort_deviation_above_band(params):
    d = params.comfort_deviation(25.0)
    assert d == pytest.approx(2.0)


def test_more_deviation_more_penalty(params):
    d1 = params.comfort_deviation(19.0)
    d2 = params.comfort_deviation(18.0)
    assert d2 > d1


def test_energy_penalty_positive_power(params):
    P = 1000.0
    penalty = -params.lambda_e * P
    assert penalty < 0.0


def test_smoothness_penalty_large_jump(params):
    u_prev, u_now = 0.0, 1.0
    penalty = -params.lambda_s * (u_now - u_prev) ** 2
    assert penalty < 0.0


def test_smoothness_penalty_zero_jump(params):
    u_prev, u_now = 0.5, 0.5
    penalty = -params.lambda_s * (u_now - u_prev) ** 2
    assert penalty == pytest.approx(0.0)