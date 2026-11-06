# AquaSentinel – Architecture

## 1. Data flow
```
                ┌──────────────────────────┐
 ScenarioConfig │ simulation.py            │  "real" network: random demand + leak + noisy sensors
 (seed, leak,  ─▶  ├─ hydraulics.py (real)  ├──▶ measured readings  (T × 9 sensors, may contain NaN)
  demand, noise)│  └─ hydraulics.py (twin)  ├──▶ expected readings  (digital twin: forecast demand, no leak)
                └──────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │ features.py              │  residual = measured − expected; NaN → 0 (twin fallback);
                │                          │  causal 1-hour rolling mean
                └──────────────────────────┘
                              │ features (T × 9)
             ┌────────────────┴────────────────┐
             ▼                                 ▼
 ┌─────────────────────┐            ┌──────────────────────────┐
 │ anomaly.py          │ alarm step │ localization.py          │
 │ Isolation Forest +  ├───────────▶│ Random Forest votes,     │
 │ threshold + 3-step  │            │ averaged since the alarm │
 │ persistence         │            │ + explanation            │
 └─────────────────────┘            └──────────────────────────┘
                                               │ predicted pipe, score
                                               ▼
           ┌───────────────────────────────────────────────────────┐
           │ pipeline.py: leak flow = extra inflow at reservoir     │
           │ meter → savings.py → prioritization.py                 │
           └───────────────────────────────────────────────────────┘
                                               │ result dict
                                               ▼
                                   app.py + visuals.py (Streamlit)
```

## 2. The simulation (and what it is not)
- **Network** (`network.py`): 1 reservoir (head 75 m), 16 junctions on a 4 × 4 grid
  (400 m spacing), 22 pipes of 80–250 mm. Only 6 junctions have pressure sensors and only 3
  pipes have flow meters – limited coverage on purpose.
- **Hydraulics** (`hydraulics.py`): Hazen-Williams head loss
  `h = 10.67·L·Q^1.852 / (C^1.852·D^4.87)` solved with the *linear theory method*: linearise
  each pipe around the current flow, solve the linear system for junction heads (NumPy),
  repeat 40 times with damping. Mass balance is checked in the tests.
- **Leaks**: pressure-dependent outflow `Q = k·√P` at the middle of a pipe (shared by its
  two end junctions). Severity 0–1 maps to k = 0.03–0.45 (≈ 0.2–2.8 L/s).
- **Demand**: daily pattern (morning/evening peaks) × demand level × random node (±8 %)
  and network-wide (±2 %) variation. The twin only knows the forecast.
- **Sensors**: Gaussian noise (0.15 m pressure, 1 % flow) × noise level, random missing
  readings, optional dead sensor.

**Honesty statement:** this is a simplified, teaching-grade steady-state model. It has *not*
been validated against EPANET or field data. It has no tanks, pumps, valves or transient
effects.

## 3. Machine learning
| | Isolation Forest (detection) | Random Forest (localisation) |
|---|---|---|
| Question | “Is today unusual?” | “Which pipe is leaking?” |
| Training data | 120 leak-free days (11 520 time steps) | 308 leak days (≈ 15 800 time steps after leak onset) |
| Labels needed | No (unsupervised) | Yes (pipe name) |
| Input | 9 standardised residual features | 9 residual features |
| Output | anomaly score → alarm | vote share per pipe |
| Key settings | 300 trees; threshold = 99.5 % quantile of scores on 40 separate calibration days; plus a range check (largest sensor |z|, same quantile × 1.5 margin); alarm after 3 consecutive anomalous steps | 300 trees, min 2 samples per leaf |

**Why the range check?** Isolation Forest isolates a point quickly only when a tree splits on
the feature where it is unusual. A leak on the trunk main P1 moved the inlet flow by ~15σ but
barely changed any pressure, so with 9 features the IF score stayed below threshold and the
leak was missed. The range check catches “one sensor far out of range”; the IF catches
“several sensors slightly off together”. Combined score = max(IF / IF threshold,
max|z| / z threshold), so 1.0 is the alarm line. The 1.5 margin was chosen on separate tuning
seeds (3001–3005): without it, false alarms reached ~80 % when sensor noise tripled. Effect on
the baseline test set: recall 0.86 → 0.94, median latency 75 → 45 min, same false alarm rate.

**No leakage between train and test:** each set is built from a different random seed
(train 100, calibration 200, tuning 3001–3005, tests 1001–1005), so no simulated day appears
in two sets. We
never split individual rows at random.

**Explanation:** contribution of each sensor = |z-score now| × global feature importance,
normalised to 100 %. We also compare the observed z-score pattern with the average
“fingerprint” of the predicted pipe (cosine similarity). This is a transparent heuristic,
not SHAP.

**Confidence:** the share of trees voting for a pipe. It ranks candidates; it is *not* a
calibrated probability. Calibration (e.g. `CalibratedClassifierCV`) is a listed improvement.

## 4. Module interfaces (contracts between team members)
| Producer → Consumer | Interface |
|---|---|
| `network.build_network()` → everyone | `WaterNetwork` dataclass: `pipe_names`, `junction_names`, `positions`, `sensor_feature_names()`, `neighbouring_pipes(pipe)`, incidence matrices |
| `network.compute_pipe_importance(net)` → prioritisation, UI | `dict[pipe] -> 0..1` |
| `simulation.simulate_scenario(net, ScenarioConfig)` → features, UI | dict: `time_hours`, `measured` (DataFrame), `expected` (DataFrame), `true_state`, `leak_active`, `true_leak_lps`, `scenario` |
| `features.scenario_features(sim)` → models | DataFrame T × 9, columns = `net.sensor_feature_names()` |
| `AnomalyDetector.detect(features)` → pipeline | dict: `scores`, `anomalous`, `alarm`, `first_alarm_step` |
| `LeakLocalizer.predict_window(features)` → pipeline | DataFrame `[pipe, score]` sorted |
| `LeakLocalizer.explain(z, pipe)` → UI | (DataFrame of sensor contributions, similarity) |
| `savings.estimate_savings(lps, cost)` → UI | dict: `daily_m3`, `monthly_m3`, `monthly_cost`, … |
| `prioritization.rank_incidents(list, importance)` → UI | ranked DataFrame |
| `pipeline.analyse_scenario(sim, detector, localizer, importance)` → UI, evaluation | result dict (status, ranking, predicted_pipe, confidence, explanation, leak_flow_lps, savings, priority, action) |

Because these contracts are fixed, each member can work in parallel and test their module
alone, e.g. Member 3 can build charts from a saved `simulate_scenario` result before the
models exist.

## 5. Team split
| Member | Owns | First milestone |
|---|---|---|
| 1 – AI/ML | `anomaly.py`, `localization.py`, `training.py`, `evaluation.py`, `train.py` | `python train.py --quick` prints metrics |
| 2 – Simulation | `network.py`, `hydraulics.py`, `simulation.py` | mass-balance test passes; leak lowers pressure |
| 3 – Dashboard | `app.py`, `visuals.py`, `.streamlit/config.toml` | network map + sensor charts render |
| 4 – Integration/Product | `features.py`, `pipeline.py`, `prioritization.py`, `savings.py`, tests, docs | `python -m pytest -q` green, demo rehearsed |

## 6. Upgrade path to WNTR / EPANET
WNTR (`pip install wntr`) wraps EPANET and is the standard Python tool for this. To upgrade:
1. Build the same network as an EPANET `.inp` file (or with `wntr.network.WaterNetworkModel`:
   `add_reservoir`, `add_junction`, `add_pipe`) – or load a public benchmark such as Net3
   or L-Town (BattLeDIM 2020).
2. Add a leak with `wn.get_node(j).add_leak(wn, area=..., start_time=...)` and run
   `wntr.sim.WNTRSimulator(wn).run_sim()` (pressure-dependent demand).
3. Replace the body of `simulation.simulate_scenario` so it returns the same dict
   (pressures from `results.node['pressure']`, flows from `results.link['flowrate']`).
   Nothing downstream needs to change – that is the point of the interface above.
4. Retrain with `python train.py` and compare the metrics with this simplified model.
We did not ship WNTR in the MVP because it adds install risk on Windows/Python versions
during a hackathon; the interface is ready for it.
