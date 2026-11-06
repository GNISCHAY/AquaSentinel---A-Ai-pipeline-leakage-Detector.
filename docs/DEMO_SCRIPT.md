# Two-minute demo script

**Before going on stage:** run `python train.py` once, start `streamlit run app.py`, set the
browser to full screen, sidebar defaults (leak in P16, severity 0.60, start 10:00, seed 2026),
“Reveal hidden ground truth” **off**, repair queue empty.

| Time | Say | Do |
|---|---|---|
| 0:00 | “This is AquaSentinel, monitoring a digital twin of a water network. Everything you see is simulated – we label it on every screen.” | Live Monitor tab |
| 0:15 | “22 pipes, but only 6 pressure sensors and 3 flow meters – like a real utility with a limited budget. Somewhere a leak has started. We don't tell the AI where.” | Point at map and KPI cards |
| 0:30 | “The Isolation Forest learned what normal days look like. The leak started at 10:00; the anomaly score stayed above 1.0 for 45 minutes straight, and the alarm came at 10:45.” | Point at anomaly timeline |
| 0:45 | “The Random Forest says pipe P16. Here's why: flow meter P5 and inlet flow are far above normal, and the pattern matches P16's learned fingerprint.” | Leak Localisation tab |
| 1:05 | “Now the reveal.” | Toggle *Reveal hidden ground truth* → white X on P16 |
| 1:15 | “Which leak first? Priority combines water loss, pipe importance – this one feeds a hospital – and confidence. Fixing it saves about 170 m³ a day.” | Repair Priority tab, click *Load 4 demo incidents* |
| 1:30 | “What if conditions are worse – more noise, missing data, a hotter day?” | What-If Lab, show Scenario B row |
| 1:45 | “We test honestly: unseen days, stress tests, and we show where it fails – small leaks and demand surges.” | Model Evaluation tab |
| 1:55 | “AquaSentinel: detect, locate, explain, prioritise. Intelligence beneath every drop.” | |

**Backup if something breaks:** screenshots of each tab in the slide deck; seed 2026 always
reproduces the same day.

---

# Feature test checklist (run before the demo)
| # | Test | Expected |
|---|---|---|
| 1 | `python -m pytest -q` | 9 passed |
| 2 | Toggle *Inject a leak* off | Status Normal (or Watch), no predicted pipe |
| 3 | Leak P16, severity 0.6 | Leak Suspected, P16 predicted |
| 4 | Severity 0.05 | Often not detected – small leaks are hard (expected) |
| 5 | Leak P20 (only feed to the south-east) | High importance, high priority |
| 6 | Noise 3.0 | Noisier anomaly score; detection usually later |
| 7 | Missing readings 25 % | Gaps in charts, status table shows ⚫ Missing |
| 8 | Failed sensor = flow_P1 | Leak-flow estimate becomes 0, explanation shifts to pressures |
| 9 | Demand 1.4, no leak | With seed 2026 this raises a false alarm (low score 0.17) – explain why (demand surge ≈ leak) |
| 9b | Leak P1, severity 1.0 | Detected and localised – the case that the range check fixed |
| 10 | Change seed | Different day; same seed returns identical results |
| 11 | Add incident + load demo incidents | Ranked queue, total savings |
| 12 | What-If Lab | Side-by-side table, before/after repair charts |
| 13 | Model Evaluation | Metrics, stress tests, confusion matrix |
