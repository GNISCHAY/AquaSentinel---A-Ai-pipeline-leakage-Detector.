"""
Train and evaluate AquaSentinel from the command line.

    python train.py            # full training + evaluation (a few minutes)
    python train.py --quick    # smaller datasets for a fast test run

Outputs (in the models/ folder):
    anomaly_detector.joblib, leak_localizer.joblib, metrics.json
"""

import argparse
import json
import time
from datetime import datetime

from aquasentinel import config
from aquasentinel.evaluation import evaluate_all
from aquasentinel.network import build_network, compute_pipe_importance
from aquasentinel.pipeline import save_models
from aquasentinel.training import train_models


def main():
    parser = argparse.ArgumentParser(description="Train AquaSentinel models")
    parser.add_argument("--quick", action="store_true", help="small datasets for a fast test")
    args = parser.parse_args()

    start = time.time()
    network = build_network()
    pipe_importance = compute_pipe_importance(network)

    if args.quick:
        detector, localizer, info = train_models(network, num_normal=30, repeats_per_pipe=4,
                                                 num_calibration=10)
        metrics = evaluate_all(network, detector, localizer, pipe_importance,
                               num_normal=10, repeats_per_pipe=1)
    else:
        detector, localizer, info = train_models(network)
        metrics = evaluate_all(network, detector, localizer, pipe_importance)

    save_models(detector, localizer, pipe_importance)
    metrics["training"] = info
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")
    metrics["quick_mode"] = args.quick
    metrics["note"] = ("All metrics are measured on SIMULATED data from a simplified "
                       "hydraulic model. They do not represent real-world performance.")
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    baseline = metrics["test_sets"]["baseline"]
    print("\n=== Baseline test set (unseen simulated days) ===")
    for key in ("precision", "recall", "f1", "false_alarm_rate_leak_free_days",
                "median_latency_minutes", "top1_accuracy", "top3_accuracy",
                "within_1_hop_accuracy"):
        print(f"  {key:35s} {baseline[key]}")
    print("\n=== Stress tests (recall / top-1 / false alarm rate) ===")
    for name, result in metrics["test_sets"].items():
        print(f"  {name:16s} recall={result['recall']}  top1={result['top1_accuracy']}  "
              f"FAR={result['false_alarm_rate_leak_free_days']}")
    print(f"\nSaved models and metrics to {config.MODELS_DIR}  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
