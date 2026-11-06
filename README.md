# 💧 AquaSentinel 💧
### Intelligent Water Network Monitoring & Predictive Leak Localization
**Intelligence Beneath Every Drop.** — Tech Wave 2026 AI Hackathon, VIT Mauritius.

> ⚠️ **Prototype on simulated data.** Every network, sensor reading, leak and saving in this 
> project is synthetic. It is a decision-support prototype, not an operational utility system.

AquaSentinel watches a digital twin of a water distribution network, detects unusual
behaviour with an **Isolation Forest**, predicts the **most likely leaking pipe** with a
**Random Forest**, explains *why*, estimates the **water lost**, and ranks **which repair
to do first**.

---

## Features
| Feature | Where in the app |
|---|---|
| Interactive digital twin (reservoir, 16 junctions, 22 pipes, 9 sensors) | 🛰️ Live Monitor |
| Synthetic pressure/flow data with demand cycles, noise, missing data, sensor failure, seeds | Sidebar |
| Isolation Forest anomaly detection + 45-min persistence rule | 🛰️ Live Monitor |
| Random Forest leak localisation, top candidates, sensor-level explanation | 🎯 Leak Localisation |
| Weighted repair priority (0.5 loss + 0.3 importance + 0.2 confidence) and repair queue | 🛠️ Repair Priority |
| Daily / monthly water and cost savings (simulated estimates) | 🛠️ Repair Priority |
| What-if comparison of two scenarios + before/after repair | 🧪 What-If Lab |
| Precision, recall, F1, confusion matrices, false alarm rate, latency, stress tests | 📊 Model Evaluation |

## Results (unseen simulated test days, from `python train.py`)
| Metric | Baseline | High noise | 15–25 % missing | Dead sensor | Unusual demand |
|---|---|---|---|---|---|
| Detection recall | 0.94 | 0.92 | 0.88 | 0.90 | 0.95 |
| False alarm rate (leak-free days) | 0.05 | 0.05 | 0.00 | 0.05 | **0.35** |
| Top-1 localisation | 0.88 | 0.78 | 0.87 | 0.59 | 0.84 |

Baseline also: precision 0.98, F1 0.96, top-3 0.95, correct pipe or neighbour 0.98,
median detection latency 45 min, leak-flow estimate error ≈ 0.05 L/s.
The weak spots are reported on purpose: small leaks (recall 0.72), a dead sensor
(top-1 drops to 0.59) and unusual demand days (35 % false alarms – a demand surge
looks like a leak). Exact numbers can change slightly if you retrain on another machine.

## Fastest way to run it
**Double-click `run.bat`** in the project folder. It finds Python, installs the libraries the
first time, starts the dashboard and opens http://localhost:8501 in your browser. Keep the black
window open while you use the app; close it to stop.

Or, in VS Code, open `app.py` and press the **▶ Run** button (same as typing `python app.py`).

## Full setup (Windows · VS Code · PowerShell)
1. Install **Python 3.10 or newer** from python.org (tick *“Add Python to PATH”*).
2. Open the project folder in VS Code → *Terminal → New Terminal* (PowerShell).
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run once:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Train and evaluate the models (≈ 3 minutes; writes `models/`):
   ```powershell
   python train.py
   ```
   Use `python train.py --quick` for a 30-second smoke run.
6. Launch the dashboard (opens http://localhost:8501):
   ```powershell
   python app.py
   ```
   (`streamlit run app.py` still works too.)
7. Run the automated tests:
   ```powershell
   python -m pytest -q
   ```

If you skip step 5 the app trains the models automatically on first launch (≈ 1 min), but the
Model Evaluation tab stays empty until you run `python train.py`.

## Project structure
```
aquasentinel-ai/
├── run.bat                   # Double-click launcher
├── app.py                    # Streamlit dashboard (Member 3); also runs with "python app.py"
├── train.py                  # Train + evaluate + save models (Member 1)
├── requirements.txt
├── .streamlit/config.toml    # Dark water-technology theme
├── aquasentinel/
│   ├── config.py             # All tunable constants in one place
│   ├── network.py            # Network graph, sensors, pipe importance (Member 2)
│   ├── hydraulics.py         # Simplified Hazen-Williams solver (Member 2)
│   ├── simulation.py         # Demand, leaks, noise, missing data (Member 2)
│   ├── features.py           # Residuals vs digital twin + smoothing (Members 1/4)
│   ├── anomaly.py            # Isolation Forest detector (Member 1)
│   ├── localization.py       # Random Forest localiser + explanation (Member 1)
│   ├── training.py           # Scenario-separated training (Member 1)
│   ├── evaluation.py         # Metrics and stress tests (Member 1)
│   ├── prioritization.py     # Repair priority formula (Member 4)
│   ├── savings.py            # Water / cost savings (Member 4)
│   ├── pipeline.py           # Glues everything together (Member 4)
│   └── visuals.py            # Plotly figures (Member 3)
├── tests/test_aquasentinel.py   # Automated tests (Member 4)
├── models/                   # Created by train.py
└── docs/                     # Architecture, pitch deck, demo script, ethics, judges' Q&A
```

## How it works (one paragraph)
A simplified hydraulic solver (Hazen-Williams head loss, linear-theory iteration) simulates
the “real” network with random customer demand, an optional pressure-dependent leak and
noisy sensors. A second run with only the *forecast* demand and no leak is the **digital
twin**. Residuals (measured − expected) remove the daily demand cycle. The Isolation Forest,
trained only on leak-free days, flags unusual combinations of residuals, and a learned
single-sensor range check covers its blind spot (one sensor far out of range); three
anomalous readings in a row raise an alarm. The Random Forest, trained on labelled simulated leaks, maps the residual
“fingerprint” since the alarm to the most likely pipe. Extra inflow at the reservoir meter
estimates the leak size, which drives savings and the priority score.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Documentation
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) – architecture, module interfaces, team split, WNTR/EPANET upgrade path
- [docs/PITCH_DECK.md](docs/PITCH_DECK.md) – 8-slide pitch deck content
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) – 2-minute demo script + feature test checklist
- [docs/IMPACT_ETHICS.md](docs/IMPACT_ETHICS.md) – one-page impact and ethics note
- [docs/JUDGES_QA.md](docs/JUDGES_QA.md) – likely judges' questions with answers

## Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Use `run.bat` or `python app.py` instead |
| `run.bat` says Python was not found | Reinstall Python and tick *“Add Python to PATH”* |
| `ModuleNotFoundError: aquasentinel` | Run commands from the project folder (the one containing `app.py`) |
| Activate.ps1 cannot be loaded | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| App shows old behaviour after code edits | Stop Streamlit (Ctrl+C) and start it again |
| Errors loading `.joblib` after upgrading scikit-learn | Delete `models/` and run `python train.py` again |
| Port 8501 busy (app already running in another window) | Close the other window, or run `streamlit run app.py --server.port 8502` |

## Limitations
Synthetic network and data; simplified solver not validated against EPANET; one leak per
simulated day; scores are not calibrated probabilities; real deployment needs real network
data, sensor validation, recalibration and expert testing.

## License
MIT – synthetic data only, no personal data.

## Build By "The Bug Dealers"
Nischay
Shauriya 
Jayesh
Sai Shree Ram
