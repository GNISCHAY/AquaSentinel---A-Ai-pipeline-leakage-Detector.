"""
Compute the headline numbers for the "We measure how well it works" slide.

    python slide_metrics.py

1. Top-3 localisation accuracy  - unseen baseline test days (same as train.py)
2. Mean distance to true leak   - distance between the middle of the predicted
                                  pipe and the middle of the leaking pipe
3. Water saved vs random order  - one crew fixes one leak per day for a week;
                                  AquaSentinel's priority order vs. random order

All numbers come from SIMULATED data.
"""

import numpy as np

from aquasentinel.evaluation import TEST_SETS, run_scenarios
from aquasentinel.network import GRID_SPACING_M, build_network
from aquasentinel.pipeline import analyse_scenario, load_models
from aquasentinel.prioritization import rank_incidents
from aquasentinel.simulation import ScenarioConfig, make_scenario_plan, simulate_scenario

SECONDS_PER_DAY = 86_400


def pipe_midpoint(network, pipe):
    k = network.pipe_index(pipe)
    (x0, y0) = network.positions[network.pipe_start[k]]
    (x1, y1) = network.positions[network.pipe_end[k]]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2]) * GRID_SPACING_M


def localisation_numbers(network, detector, localizer, importance):
    seed, overrides, _ = TEST_SETS["baseline"]
    plan = make_scenario_plan(network, seed, 40, 4, **overrides)
    records = run_scenarios(network, plan, detector, localizer, importance)
    localised = records[records["top1"].notna()]
    distances = [np.linalg.norm(pipe_midpoint(network, row.predicted_pipe) - pipe_midpoint(network, row.true_pipe))
                 for row in localised.itertuples()]
    return float(localised["top3"].mean()), float(np.mean(distances)), float(np.median(distances)), len(localised)


def repair_order_numbers(network, detector, localizer, importance, trials=300, leaks_per_week=7):
    """A week with 7 detected leaks; the crew repairs one per day."""
    rng = np.random.default_rng(7)

    # Pool of simulated detected leaks (seeds 5000+ are not used anywhere else)
    pool = []
    for i in range(160):
        scenario = ScenarioConfig(seed=5000 + i, leak_pipe=str(rng.choice(network.pipe_names)),
                                  severity=float(rng.uniform(0.05, 1.0)), leak_start_step=40)
        simulation = simulate_scenario(network, scenario)
        result = analyse_scenario(simulation, detector, localizer, importance)
        if result["status"] == "LEAK SUSPECTED":
            pool.append({"pipe": result["predicted_pipe"], "leak_flow_lps": result["leak_flow_lps"],
                         "confidence": result["confidence"],
                         "true_flow_lps": float(simulation["true_leak_lps"][40:].mean())})

    def litres_lost(order):
        # The leak repaired on day d keeps leaking for d days (d = 1..7)
        return sum(item["true_flow_lps"] * SECONDS_PER_DAY * day for day, item in enumerate(order, start=1))

    saved = []
    for _ in range(trials):
        week = [dict(pool[i], incident_id=str(n)) for n, i in
                enumerate(rng.choice(len(pool), leaks_per_week, replace=False))]
        ranked = rank_incidents(week, importance)
        smart_order = [week[int(i)] for i in ranked["incident_id"]]
        random_loss = np.mean([litres_lost([week[i] for i in rng.permutation(leaks_per_week)]) for _ in range(50)])
        saved.append(random_loss - litres_lost(smart_order))
    saved_m3_week = np.mean(saved) / 1000
    return saved_m3_week, saved_m3_week / leaks_per_week, len(pool)


def main():
    network = build_network()
    detector, localizer, importance = load_models()
    top3, mean_dist, median_dist, days = localisation_numbers(network, detector, localizer, importance)
    saved_week, saved_day, pool_size = repair_order_numbers(network, detector, localizer, importance)
    print("\n=== Slide numbers (simulated data) ===")
    print(f"Top-3 localisation accuracy : {top3:.0%}   ({days} detected leak days)")
    print(f"Mean distance to true leak  : {mean_dist:.0f} m  (median {median_dist:.0f} m)")
    print(f"Water saved vs random order : {saved_day:.0f} m3/day  ({saved_week:.0f} m3 over the first week, "
          f"7 leaks, 300 simulated weeks, pool of {pool_size} detected leaks)")


if __name__ == "__main__":
    main()
