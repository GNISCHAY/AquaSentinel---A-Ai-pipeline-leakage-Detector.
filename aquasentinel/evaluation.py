"""
Model evaluation (Member 1 - AI/ML Developer).

Evaluation is done per SCENARIO (per simulated day), end-to-end, using the
same `analyse_scenario` pipeline as the dashboard:

Detection (scenario level)
  * TP = leak day where an alarm is raised after the leak starts
  * FN = leak day with no alarm after the leak starts
  * FP = leak-free day with any alarm
  * TN = leak-free day with no alarm
  -> precision, recall, F1, false alarm rate, detection latency

Localisation (only on detected leak days)
  * top-1 accuracy  : predicted pipe is the true pipe
  * top-3 accuracy  : true pipe is in the 3 highest-scoring pipes
  * within-1-hop    : prediction is the true pipe or a pipe touching it
                      (a crew searching there would find the leak)

Stress tests repeat everything under harder conditions than training.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from aquasentinel import config
from aquasentinel.pipeline import analyse_scenario
from aquasentinel.simulation import ScenarioConfig, make_scenario_plan, simulate_scenario

# name -> (seed, extra arguments for make_scenario_plan, description)
TEST_SETS = {
    "baseline": (1001, {}, "Same conditions as training, unseen days"),
    "high_noise": (1002, {"noise_range": (2.5, 3.0)}, "Sensor noise 2.5-3x typical"),
    "missing_data": (1003, {"missing_range": (0.15, 0.25)}, "15-25% of readings lost"),
    "sensor_failure": (1004, {"sensor_failure_prob": 1.0}, "One sensor dead all day"),
    "unusual_demand": (1005, {"demand_range": (1.25, 1.4)}, "Demand 25-40% above normal (outside training range)"),
}
SEVERITY_BANDS = [("small", 0.0, 0.3), ("medium", 0.3, 0.6), ("large", 0.6, 1.01)]


def severity_band(severity):
    for name, low, high in SEVERITY_BANDS:
        if low <= severity < high:
            return name
    return "large"


def run_scenarios(network, plan, detector, localizer, pipe_importance):
    """Run the full pipeline on every scenario and record one row per day."""
    records = []
    for scenario in plan:
        simulation = simulate_scenario(network, scenario)
        result = analyse_scenario(simulation, detector, localizer, pipe_importance)
        alarm = result["detection"]["alarm"]
        is_leak = scenario.leak_pipe is not None
        onset = scenario.leak_start_step if is_leak else len(alarm)

        alarm_before_onset = bool(alarm[:onset].any())
        alarm_after_onset = bool(alarm[onset:].any()) if is_leak else False
        latency_steps = int(np.argmax(alarm[onset:])) if alarm_after_onset else None

        record = {
            "is_leak": is_leak,
            "true_pipe": scenario.leak_pipe or "none",
            "severity": scenario.severity if is_leak else 0.0,
            "severity_band": severity_band(scenario.severity) if is_leak else "none",
            "detected": alarm_after_onset,
            "false_alarm": alarm_before_onset if is_leak else bool(alarm.any()),
            "latency_minutes": None if latency_steps is None else latency_steps * config.TIMESTEP_MINUTES,
            "predicted_pipe": None,
            "top1": None, "top3": None, "within_1_hop": None,
            "true_leak_lps": float(simulation["true_leak_lps"][onset:].mean()) if is_leak else 0.0,
            "estimated_leak_lps": result["leak_flow_lps"],
        }
        # Only score localisation if the alarm came after the true onset
        if is_leak and alarm_after_onset and not alarm_before_onset:
            ranking = result["ranking"]
            predicted = result["predicted_pipe"]
            neighbours = network.neighbouring_pipes(scenario.leak_pipe)
            record.update({
                "predicted_pipe": predicted,
                "top1": predicted == scenario.leak_pipe,
                "top3": scenario.leak_pipe in ranking["pipe"].head(3).tolist(),
                "within_1_hop": predicted == scenario.leak_pipe or predicted in neighbours,
            })
        records.append(record)
    return pd.DataFrame(records)


def summarise(records):
    """Turn per-day records into the headline metrics."""
    y_true = records["is_leak"].astype(int)
    # A day "raises an alarm" if it is a detected leak or has any false alarm
    y_pred = (records["detected"] | records["false_alarm"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    normal = records[~records["is_leak"]]
    leaks = records[records["is_leak"]]
    localised = leaks[leaks["top1"].notna()]
    latencies = leaks["latency_minutes"].dropna()

    summary = {
        "days_tested": int(len(records)),
        "leak_days": int(len(leaks)),
        "leak_free_days": int(len(normal)),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "false_alarm_rate_leak_free_days": round(float(normal["false_alarm"].mean()), 3) if len(normal) else None,
        "leak_days_with_early_false_alarm": int(leaks["false_alarm"].sum()),
        "median_latency_minutes": float(latencies.median()) if len(latencies) else None,
        "mean_latency_minutes": round(float(latencies.mean()), 1) if len(latencies) else None,
        "detection_confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "localised_days": int(len(localised)),
        "top1_accuracy": round(float(localised["top1"].mean()), 3) if len(localised) else None,
        "top3_accuracy": round(float(localised["top3"].mean()), 3) if len(localised) else None,
        "within_1_hop_accuracy": round(float(localised["within_1_hop"].mean()), 3) if len(localised) else None,
    }
    recall_by_band = {}
    for name, _, _ in SEVERITY_BANDS:
        band = leaks[leaks["severity_band"] == name]
        if len(band):
            band_localised = band[band["top1"].notna()]
            recall_by_band[name] = {
                "leak_days": int(len(band)),
                "recall": round(float(band["detected"].mean()), 3),
                "top1_accuracy": round(float(band_localised["top1"].mean()), 3) if len(band_localised) else None,
            }
    summary["by_severity"] = recall_by_band
    if len(leaks[leaks["detected"]]):
        detected = leaks[leaks["detected"]]
        error = (detected["estimated_leak_lps"] - detected["true_leak_lps"]).abs()
        summary["leak_flow_mean_abs_error_lps"] = round(float(error.mean()), 3)
    return summary


def localisation_confusion(records, pipe_names):
    localised = records[records["top1"].notna()]
    matrix = confusion_matrix(localised["true_pipe"], localised["predicted_pipe"], labels=pipe_names)
    return {"labels": pipe_names, "matrix": matrix.tolist()}


def evaluate_all(network, detector, localizer, pipe_importance, num_normal=40,
                 repeats_per_pipe=4, progress=print):
    """Run every test set. Returns a metrics dict (saved to metrics.json)."""
    results = {}
    baseline_records = None
    for name, (seed, overrides, description) in TEST_SETS.items():
        progress(f"Evaluating test set '{name}' ...")
        plan = make_scenario_plan(network, seed, num_normal, repeats_per_pipe, **overrides)
        records = run_scenarios(network, plan, detector, localizer, pipe_importance)
        results[name] = {"description": description, **summarise(records)}
        if name == "baseline":
            baseline_records = records
    return {
        "test_sets": results,
        "localisation_confusion": localisation_confusion(baseline_records, network.pipe_names),
        "feature_importance": dict(zip(localizer.feature_names,
                                       np.round(localizer.model.feature_importances_, 4).tolist())),
    }


def quick_demo_check(network, detector, localizer, pipe_importance):
    """Tiny sanity check used by tests: one leak day and one normal day."""
    leak = simulate_scenario(network, ScenarioConfig(seed=7, leak_pipe="P16", severity=0.8))
    normal = simulate_scenario(network, ScenarioConfig(seed=7))
    return (analyse_scenario(leak, detector, localizer, pipe_importance),
            analyse_scenario(normal, detector, localizer, pipe_importance))
