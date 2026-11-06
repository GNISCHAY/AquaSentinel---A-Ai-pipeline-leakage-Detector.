# AquaSentinel – Pitch Deck (8 slides)

---
## Slide 1 – Title
**AquaSentinel** – Intelligence Beneath Every Drop.
AI that finds water leaks, explains them, and tells crews which one to fix first.
Team names · Tech Wave 2026 AI Hackathon, Mauritius
*Visual: dashboard screenshot with the red predicted pipe.*

---
## Slide 2 – The problem
- Water utilities lose a large share of treated water before it reaches customers
  (“non-revenue water”) – leaks are a major cause.
- Sensors are few and noisy, so utilities often know *that* water is lost, not *where*.
- Crews search by hand; small leaks run for weeks.
*Speaker note: cite a source you have checked (e.g. a World Bank or national utility
report) – do not quote numbers you cannot reference.*

---
## Slide 3 – Our solution
A digital twin + two AI models + an impact engine:
1. **Detect** – Isolation Forest + a learned range check spot unusual pressure/flow behaviour
   (thresholds learned from normal days, not hand-set).
2. **Locate** – Random Forest predicts the most likely leaking pipe.
3. **Explain** – which sensors, how far from normal, fingerprint match.
4. **Prioritise** – water loss × network importance × confidence.
5. **Quantify** – estimated m³ and money saved per day/month.

---
## Slide 4 – How it works
*Visual: the data-flow diagram from ARCHITECTURE.md.*
Measured readings − digital twin expectation = residual “fingerprint”
→ anomaly score → alarm (45 min persistence) → pipe ranking → priority & savings.
Simulation: Hazen-Williams hydraulic solver, pressure-dependent leaks, daily demand cycles,
noise, missing data, sensor failure – all reproducible with seeds.

---
## Slide 5 – Live demo
*(switch to the app – see DEMO_SCRIPT.md)*

---
## Slide 6 – Evidence (honest evaluation)
Unseen simulated test days, separated from training by scenario:
- Detection: precision 0.98 · recall 0.94 · F1 0.96 · median latency 45 min
- Localisation: exact pipe 88 % · top-3 95 % · pipe-or-neighbour 98 %
- Stress tests: high noise, 25 % missing data, dead sensor, unusual demand
- **Where it struggles:** small leaks (72 % recall), dead sensor (59 % exact pipe),
  unusual-demand days (35 % false alarms)
- We found and fixed a real blind spot: Isolation Forest missed trunk-main leaks → added a
  range check (recall 0.86 → 0.94)

---
## Slide 7 – Impact & responsibility
- A 2 L/s leak ≈ 170 m³/day ≈ 5 000 m³/month (simulated estimate).
- Faster, targeted repairs → less water lost, less energy pumping it, fewer road excavations.
- Synthetic data only, no personal data; decision support – engineers decide.
- Transparent priority weights a utility can review with stakeholders.

---
## Slide 8 – Roadmap & ask
1. Swap the simplified solver for WNTR/EPANET and a public benchmark (L-Town / BattLeDIM).
2. Pilot on a real district metered area with historical data from a utility partner.
3. Optimal sensor placement, probability calibration, multiple simultaneous leaks.
**Ask:** access to an anonymised network model and sensor data for a pilot study.
