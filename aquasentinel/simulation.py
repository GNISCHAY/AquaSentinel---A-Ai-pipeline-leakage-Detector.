"""
Synthetic data generation and leak simulation (Member 2 - Simulation Developer).

One "scenario" = one simulated day (96 readings, every 15 minutes).

For every scenario we produce TWO versions of the network:
  * the "real" network  -> random customer demand + optional leak + noisy sensors
  * the "digital twin"  -> what the utility EXPECTS: forecast demand, no leak

The difference between the two (the residual) is what the AI analyses.
ALL VALUES PRODUCED HERE ARE SIMULATED.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from aquasentinel import config
from aquasentinel.hydraulics import solve_hydraulics


@dataclass
class ScenarioConfig:
    """All the knobs of one simulated day. Same seed -> same results."""

    seed: int = config.GLOBAL_SEED
    leak_pipe: str = None            # None = no leak
    severity: float = 0.5            # 0 (tiny) .. 1 (burst)
    leak_start_step: int = 40        # 40 x 15 min = 10:00
    demand_level: float = 1.0        # 1.0 = normal day, 1.3 = hot day
    noise_level: float = 1.0         # multiplies the sensor noise
    missing_rate: float = 0.0        # fraction of readings randomly lost
    failed_sensor: str = None        # one sensor channel dead all day

    def to_dict(self):
        return asdict(self)


def diurnal_pattern(num_steps=config.STEPS_PER_DAY):
    """
    Typical daily demand shape: low at night, peaks in the morning and evening.
    Returns one multiplier per time step (average roughly 1.0).
    """
    hours = np.arange(num_steps) * config.TIMESTEP_MINUTES / 60.0

    def bump(centre, width):
        return np.exp(-0.5 * ((hours - centre) / width) ** 2)

    pattern = 0.45 + 0.75 * bump(7.5, 1.5) + 0.35 * bump(13.0, 2.0) + 0.65 * bump(19.5, 2.0)
    return pattern / pattern.mean()


def severity_to_coefficient(severity):
    """Map the 0..1 severity slider onto the physical leak coefficient."""
    severity = float(np.clip(severity, 0.0, 1.0))
    return config.LEAK_COEFFICIENT_MIN + severity * (
        config.LEAK_COEFFICIENT_MAX - config.LEAK_COEFFICIENT_MIN
    )


def _sensor_readings(network, hydraulic_result):
    """Pick the sensor channels out of a full hydraulic solution."""
    columns = {}
    for node in network.pressure_sensor_nodes:
        columns[f"pressure_{node}"] = hydraulic_result["pressures_m"][:, network.junction_index(node)]
    for pipe in network.flow_meter_pipes:
        columns[f"flow_{pipe}"] = hydraulic_result["flows_lps"][:, network.pipe_index(pipe)]
    return pd.DataFrame(columns)


def simulate_scenario(network, scenario):
    """
    Run one simulated day.

    Returns a dict with:
        time_hours      array (T,)
        measured        DataFrame (T, sensors) - noisy, may contain NaN
        expected        DataFrame (T, sensors) - digital twin prediction
        true_state      dict of full hydraulic arrays of the "real" network
        leak_active     bool array (T,) - ground truth (hidden from the AI)
        true_leak_lps   array (T,)      - ground truth leak flow
        scenario        the ScenarioConfig used
    """
    rng = np.random.default_rng(scenario.seed)
    steps = config.STEPS_PER_DAY
    time_hours = np.arange(steps) * config.TIMESTEP_MINUTES / 60.0
    pattern = diurnal_pattern(steps)

    # --- Demand: forecast (twin) vs. actual (real network) -----------------
    # The random draws always happen in the same order, so a scenario with and
    # without a leak share exactly the same demand and noise (fair comparison).
    forecast_demand = np.outer(pattern * scenario.demand_level, network.base_demands_lps)
    global_drift = 1.0 + rng.normal(0, config.GLOBAL_DEMAND_VARIATION, size=(steps, 1))
    node_variation = 1.0 + rng.normal(0, config.NODE_DEMAND_VARIATION, size=forecast_demand.shape)
    actual_demand = np.clip(forecast_demand * global_drift * node_variation, 0.0, None)

    # --- Leak --------------------------------------------------------------
    leak_coefficients = np.zeros((steps, network.num_pipes))
    leak_active = np.zeros(steps, dtype=bool)
    if scenario.leak_pipe is not None:
        leak_active[scenario.leak_start_step:] = True
        k = network.pipe_index(scenario.leak_pipe)
        leak_coefficients[leak_active, k] = severity_to_coefficient(scenario.severity)

    # --- Hydraulics --------------------------------------------------------
    true_state = solve_hydraulics(network, actual_demand, leak_coefficients)
    twin_state = solve_hydraulics(network, forecast_demand)
    clean = _sensor_readings(network, true_state)
    expected = _sensor_readings(network, twin_state)

    # --- Sensor noise and data quality problems ---------------------------
    measured = clean.copy()
    for column in measured.columns:
        if column.startswith("pressure_"):
            noise = rng.normal(0, config.PRESSURE_NOISE_STD_M * scenario.noise_level, steps)
        else:
            noise = rng.normal(0, config.FLOW_NOISE_FRACTION * scenario.noise_level, steps) * np.abs(clean[column])
        measured[column] = clean[column] + noise

    missing_mask = rng.random(measured.shape) < scenario.missing_rate
    measured = measured.mask(missing_mask)
    if scenario.failed_sensor is not None and scenario.failed_sensor in measured.columns:
        measured[scenario.failed_sensor] = np.nan

    true_leak_lps = true_state["leak_flows_lps"].sum(axis=1)
    return {
        "time_hours": time_hours,
        "measured": measured,
        "expected": expected,
        "true_state": true_state,
        "leak_active": leak_active,
        "true_leak_lps": true_leak_lps,
        "scenario": scenario,
    }


def make_scenario_plan(network, seed, num_normal, repeats_per_pipe,
                       noise_range=(0.5, 1.5), demand_range=(0.85, 1.15),
                       missing_range=(0.0, 0.05), severity_range=(0.05, 1.0),
                       sensor_failure_prob=0.0):
    """
    Build a list of ScenarioConfig objects covering:
      * num_normal leak-free days
      * repeats_per_pipe leak days for EVERY pipe, with random severity,
        start time, demand level, noise and missing data.
    Different `seed` values give completely different scenario sets, which is
    how we keep training and testing scenarios separate.
    """
    rng = np.random.default_rng(seed)
    channels = network.sensor_feature_names()
    plan = []

    def random_conditions():
        failed = None
        if rng.random() < sensor_failure_prob:
            failed = str(rng.choice(channels))
        return dict(
            seed=int(rng.integers(0, 2**31 - 1)),
            demand_level=float(rng.uniform(*demand_range)),
            noise_level=float(rng.uniform(*noise_range)),
            missing_rate=float(rng.uniform(*missing_range)),
            failed_sensor=failed,
        )

    for _ in range(num_normal):
        plan.append(ScenarioConfig(leak_pipe=None, **random_conditions()))

    for pipe in network.pipe_names:
        for _ in range(repeats_per_pipe):
            plan.append(ScenarioConfig(
                leak_pipe=pipe,
                severity=float(rng.uniform(*severity_range)),
                leak_start_step=int(rng.integers(24, 61)),   # between 06:00 and 15:00
                **random_conditions(),
            ))
    return plan
