"""
Sensor data processing (shared by Member 1 and Member 4).

Turns raw (simulated) sensor readings into the features used by both models:

1. Residual  = measured reading - digital twin's expected reading
   (removes the normal daily demand cycle, so only "surprises" remain)
2. Missing readings are replaced with residual 0, i.e. we fall back to the
   digital twin's expected value for that reading.
3. Smoothing = causal rolling mean over the last hour (only past data is
   used, so it could run live).
"""

import numpy as np
import pandas as pd

from aquasentinel import config
from aquasentinel.simulation import simulate_scenario

INLET_FLOW_FEATURE = "flow_P1"   # total inflow meter at the reservoir outlet
META_COLUMNS = ["scenario_id", "step", "leak_pipe", "severity",
                "leak_active", "true_leak_lps", "leak_start_step"]


def compute_residuals(measured, expected):
    """Residual of every sensor channel; missing readings -> 0 (twin fallback)."""
    residuals = measured - expected
    return residuals.fillna(0.0)


def smooth_residuals(residuals, window=config.ROLLING_WINDOW_STEPS):
    """Causal rolling mean: the value at step t only uses steps t-window+1..t."""
    return residuals.rolling(window=window, min_periods=1).mean()


def scenario_features(simulation):
    """Smoothed residual features for one simulated day (DataFrame, T x sensors)."""
    residuals = compute_residuals(simulation["measured"], simulation["expected"])
    return smooth_residuals(residuals)


def build_feature_table(simulation, scenario_id):
    """Features plus ground-truth labels for every time step of one scenario."""
    features = scenario_features(simulation)
    scenario = simulation["scenario"]
    table = features.copy()
    table["scenario_id"] = scenario_id
    table["step"] = np.arange(len(table))
    table["leak_pipe"] = scenario.leak_pipe if scenario.leak_pipe else "none"
    table["severity"] = scenario.severity if scenario.leak_pipe else 0.0
    table["leak_active"] = simulation["leak_active"]
    table["true_leak_lps"] = simulation["true_leak_lps"]
    table["leak_start_step"] = scenario.leak_start_step if scenario.leak_pipe else -1
    return table


def generate_dataset(network, plan, first_scenario_id=0):
    """Simulate every scenario in `plan` and stack their feature tables."""
    tables = []
    for offset, scenario in enumerate(plan):
        simulation = simulate_scenario(network, scenario)
        tables.append(build_feature_table(simulation, first_scenario_id + offset))
    return pd.concat(tables, ignore_index=True)


def feature_columns(network):
    """The ordered list of model input columns."""
    return network.sensor_feature_names()
