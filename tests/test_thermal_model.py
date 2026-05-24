"""Unit tests for the RC thermal model."""

import math
import pytest
import numpy as np

from smart_hvac.core.parameters import Parameters
from smart_hvac.core.thermal_model import step_rc, occupancy_schedule


@pytest.fixture
def params():
    return Parameters()


def test_no_hvac_approaches_outdoor(params):
    """Without HVAC, T_in should converge toward T_out over many steps."""
    T_out = 5.0
    T_in = 22.0
    for _ in range(500):
        T_in = step_rc(T_in=T_in, T_out=T_out, P_hvac=0.0, Q_int=0.0, params=params)
    assert T_in == pytest.approx(T_out, abs=0.5)


def test_heating_raises_temperature(params):
    """Full HVAC power should raise T_in above initial value."""
    T_in_0 = 15.0
    T_in = T_in_0
    for _ in range(20):
        T_in = step_rc(T_in=T_in, T_out=5.0, P_hvac=params.P_max, Q_int=0.0, params=params)
    assert T_in > T_in_0


def test_steady_state_with_constant_inputs(params):
    """T_in should stabilise near the analytical steady-state value."""
    T_out = 10.0
    P = params.P_max * 0.5
    T_ss = T_out + params.R * (params.eta_hvac * P)
    T_in = 22.0
    for _ in range(1000):
        T_in = step_rc(T_in=T_in, T_out=T_out, P_hvac=P, Q_int=0.0, params=params)
    assert T_in == pytest.approx(T_ss, abs=0.1)


def test_occupancy_schedule_office(params):
    rng = np.random.default_rng(0)
    occ = occupancy_schedule(params.episode_steps, params.dt, "office", rng=None)
    steps_per_hour = int(3600 / params.dt)
    # Should be occupied between 08:00 and 18:00
    assert occ[8 * steps_per_hour] == 1
    assert occ[17 * steps_per_hour] == 1
    assert occ[7 * steps_per_hour - 1] == 0


def test_occupancy_schedule_always_on(params):
    occ = occupancy_schedule(params.episode_steps, params.dt, "always_on")
    assert np.all(occ == 1)