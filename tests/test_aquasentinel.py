"""
Automated checks (Member 4 - Integration & Testing).

Run from the project folder:   python -m pytest -q
These tests use small, fast models so they finish in well under a minute.
"""

import numpy as np
import pytest

from aquasentinel.features import scenario_features
from aquasentinel.hydraulics import solve_hydraulics
from aquasentinel.network import build_network, compute_pipe_importance
from aquasentinel.pipeline import analyse_scenario
from aquasentinel.prioritization import rank_incidents
from aquasentinel.savings import estimate_savings
from aquasentinel.simulation import ScenarioConfig, simulate_scenario
from aquasentinel.training import train_models


@pytest.fixture(scope="module")
def network():
    return build_network()


@pytest.fixture(scope="module")
def trained(network):
    detector, localizer, _ = train_models(network, num_normal=25, repeats_per_pipe=4,
                                          num_calibration=8, progress=lambda message: None)
    return detector, localizer, compute_pipe_importance(network)


def test_network_shape(network):
    assert network.num_junctions == 16
    assert network.num_pipes == 22
    assert len(network.sensor_feature_names()) == 9


def test_mass_balance(network):
    """Water entering from the reservoir must equal total demand + leak."""
    demands = network.base_demands_lps[np.newaxis, :] * 1.2
    leak = np.zeros((1, network.num_pipes))
    leak[0, network.pipe_index("P9")] = 0.3
    result = solve_hydraulics(network, demands, leak)
    total_out = demands.sum() + result["leak_flows_lps"].sum()
    assert result["inflow_lps"][0] == pytest.approx(total_out, rel=1e-3)
    assert (result["pressures_m"] > 0).all()


def test_leak_lowers_pressure(network):
    base = simulate_scenario(network, ScenarioConfig(seed=1, noise_level=0.0))
    leak = simulate_scenario(network, ScenarioConfig(seed=1, noise_level=0.0,
                                                     leak_pipe="P9", severity=0.8))
    after = base["leak_active"].size - 1
    assert leak["true_state"]["pressures_m"][after].mean() < base["true_state"]["pressures_m"][after].mean()
    assert leak["true_leak_lps"][-1] > 0.5


def test_same_seed_is_reproducible(network):
    a = simulate_scenario(network, ScenarioConfig(seed=5, leak_pipe="P4", missing_rate=0.1))
    b = simulate_scenario(network, ScenarioConfig(seed=5, leak_pipe="P4", missing_rate=0.1))
    assert a["measured"].equals(b["measured"])


def test_missing_and_failed_sensors_are_handled(network):
    sim = simulate_scenario(network, ScenarioConfig(seed=3, missing_rate=0.3,
                                                    failed_sensor="pressure_J11"))
    assert sim["measured"]["pressure_J11"].isna().all()
    features = scenario_features(sim)
    assert not features.isna().any().any()


def test_pipeline_detects_large_leak(network, trained):
    detector, localizer, importance = trained
    sim = simulate_scenario(network, ScenarioConfig(seed=11, leak_pipe="P16", severity=0.9))
    result = analyse_scenario(sim, detector, localizer, importance)
    assert result["status"] == "LEAK SUSPECTED"
    assert result["detection"]["first_alarm_step"] >= sim["scenario"].leak_start_step
    assert result["predicted_pipe"] in network.pipe_names
    assert 0 <= result["priority"] <= 1
    assert result["savings"]["daily_m3"] > 0


def test_pipeline_quiet_on_normal_day(network, trained):
    detector, localizer, importance = trained
    sim = simulate_scenario(network, ScenarioConfig(seed=12))
    result = analyse_scenario(sim, detector, localizer, importance)
    assert result["status"] in ("NORMAL", "WATCH")
    assert result["savings"] is None


def test_savings_units():
    savings = estimate_savings(1.0, cost_per_m3=10)
    assert savings["daily_m3"] == pytest.approx(86.4)
    assert savings["monthly_m3"] == pytest.approx(2592)
    assert savings["monthly_cost"] == pytest.approx(25920)


def test_priority_ranking_orders_by_score():
    incidents = [
        {"incident_id": "A", "pipe": "P4", "leak_flow_lps": 0.3, "confidence": 0.5},
        {"incident_id": "B", "pipe": "P6", "leak_flow_lps": 2.5, "confidence": 0.9},
    ]
    table = rank_incidents(incidents, {"P4": 0.1, "P6": 0.7})
    assert table.loc[0, "incident_id"] == "B"
    assert list(table["rank"]) == [1, 2]
