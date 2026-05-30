"""Unit tests for the RC thermal model."""
import pytest
import numpy as np
from smart_hvac.core.parameters import Parameters
from smart_hvac.core.thermal_model import step_rc, SinusoidalWeather, occupancy_flag


@pytest.fixture
def params():
    return Parameters()


def test_cooling_toward_T_out(params):
    """Without HVAC and gains, T_in must converge toward T_out."""
    T_in, T_out = 25.0, 10.0
    for _ in range(500):
        T_in = step_rc(T_in, T_out, P_hvac=0.0, Q_int=0.0, params=params)
    assert abs(T_in - T_out) < 0.1, f"T_in={T_in:.2f} should approach T_out={T_out}"


def test_heating_increases_T_in(params):
    """With constant heating, T_in must increase."""
    T_in = 15.0
    T_out = 5.0
    T_new = step_rc(T_in, T_out, P_hvac=params.P_max, Q_int=0.0, params=params)
    assert T_new > T_in


def test_steady_state(params):
    """At steady state: T_ss = T_out + R * P_hvac."""
    P_hvac = 3000.0
    T_out  = 10.0
    T_ss_expected = T_out + params.R * params.eta_hvac * P_hvac
    T_in = T_ss_expected
    T_new = step_rc(T_in, T_out, P_hvac, Q_int=0.0, params=params)
    assert abs(T_new - T_in) < 1e-6, "At steady state, T_in should remain unchanged."


def test_weather_cold(params):
    rng = np.random.default_rng(0)
    w = SinusoidalWeather(params, rng)
    w.reset("cold")
    T = w.get(0)
    assert T < 10.0, "Cold day should give low outdoor temperature."


def test_occupancy_office(params):
    assert occupancy_flag(32, "office", params.dt) == 1  # step 32 = 8h
    assert occupancy_flag(0,  "office", params.dt) == 0  # midnight
