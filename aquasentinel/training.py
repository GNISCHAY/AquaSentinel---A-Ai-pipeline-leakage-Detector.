"""
Model training (Member 1 - AI/ML Developer).

Training and testing use DIFFERENT scenario seeds, so no simulated day ever
appears in both sets (we never split individual sensor rows at random).

    seed 100  -> training days        (Isolation Forest + Random Forest)
    seed 200  -> calibration days     (alarm threshold only, leak-free)
    seed 1000+ -> test days           (see evaluation.py)
"""

from aquasentinel import config
from aquasentinel.anomaly import AnomalyDetector
from aquasentinel.features import feature_columns, generate_dataset
from aquasentinel.localization import LeakLocalizer
from aquasentinel.simulation import make_scenario_plan

TRAIN_SEED = 100
CALIBRATION_SEED = 200


def train_models(network, num_normal=120, repeats_per_pipe=14, num_calibration=40,
                 progress=print):
    """Simulate training data and fit both models. Returns (detector, localizer, info)."""
    names = feature_columns(network)

    progress("Simulating training scenarios ...")
    train_plan = make_scenario_plan(network, TRAIN_SEED, num_normal, repeats_per_pipe,
                                    sensor_failure_prob=0.1)
    train = generate_dataset(network, train_plan, first_scenario_id=0)

    progress("Simulating leak-free calibration days ...")
    calibration_plan = make_scenario_plan(network, CALIBRATION_SEED, num_calibration, 0)
    calibration = generate_dataset(network, calibration_plan, first_scenario_id=100_000)

    progress("Training Isolation Forest on leak-free days ...")
    normal_rows = train[train["leak_pipe"] == "none"]
    detector = AnomalyDetector(names, seed=config.GLOBAL_SEED).fit(normal_rows, calibration)

    progress("Training Random Forest on labelled leak periods ...")
    # Only use steps where the 1-hour smoothing window lies fully inside the leak
    settled = train["step"] >= train["leak_start_step"] + config.ROLLING_WINDOW_STEPS - 1
    leak_rows = train[train["leak_active"] & settled]
    localizer = LeakLocalizer(names, seed=config.GLOBAL_SEED).fit(
        leak_rows, leak_rows["leak_pipe"].to_numpy(), detector.z_scores(leak_rows))

    info = {
        "train_scenarios": len(train_plan),
        "train_normal_scenarios": num_normal,
        "train_leak_scenarios": len(train_plan) - num_normal,
        "calibration_scenarios": len(calibration_plan),
        "isolation_forest_rows": int(len(normal_rows)),
        "random_forest_rows": int(len(leak_rows)),
        "isolation_forest_threshold": round(detector.if_threshold, 4),
        "range_check_threshold_z": round(detector.z_threshold, 2),
    }
    return detector, localizer, info
