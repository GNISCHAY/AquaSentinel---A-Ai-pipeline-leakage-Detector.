# Likely judges' questions – and strong answers

**1. Is this real data?**
No. Everything is simulated, and we label it everywhere. We chose synthetic data because
real network data is sensitive and not public, and because simulation gives us ground
truth to measure accuracy. The architecture accepts real data: only the simulation module
would change.

**2. How realistic is your simulation?**
It uses real hydraulic equations (Hazen-Williams head loss, a standard iterative solver,
pressure-dependent leaks) but it is simplified: no tanks, pumps or valves, and it is not
validated against EPANET. Our upgrade path is WNTR, which wraps EPANET; the interface is
already designed for it.

**3. Why Isolation Forest and not a fixed threshold?**
A fixed threshold on one sensor can't see combinations – a small drop on three sensors plus a
small rise in inflow. Isolation Forest learns the normal joint behaviour of all sensors
without needing leak labels, which matters because real utilities have few labelled leaks.
We still add a simple persistence rule (3 readings in a row) to cut false alarms.

**3b. Isolation Forest alone wasn't enough?**
Correct – we found a blind spot during testing. A leak on the trunk main pushed the inlet flow
15 standard deviations above normal but barely changed pressures; with 9 features, random
trees rarely split on that one sensor, so the score stayed low. We added a range check whose
threshold is also learned from normal days (with a 1.5× margin chosen on separate tuning
seeds). Recall rose from 0.86 to 0.94 and latency fell from 75 to 45 min, with the same false
alarm rate. Isolation Forest still does the part it is good at: unusual combinations.

**4. Why compare with a digital twin instead of using raw readings?**
Raw pressure changes all day with demand. Subtracting what the twin expects removes the
daily cycle, so the models see only surprises.

**5. Why Random Forest for localisation?**
It handles non-linear patterns, needs little tuning, works with small tabular data, gives
feature importances for explanation, and trains in seconds on a laptop.

**6. How did you avoid data leakage?**
Training, calibration and test sets are generated from different random seeds – whole
simulated days, never randomly split rows. The alarm threshold is set on separate leak-free
days, not on training data.

**7. Your accuracy looks high – is it overfitted?**
It's measured on unseen days, but those days come from the same simulator, so it is
optimistic for the real world. That's why we run stress tests and publish the weak spots:
small leaks, dead sensors, unusual demand.

**8. What is the “confidence” number?**
The share of trees voting for that pipe. It ranks candidates; it is not a calibrated
probability. We'd calibrate it with real data before using it as a probability.

**9. What happens when a sensor fails?**
Missing readings fall back to the twin's expected value, so the pipeline keeps running.
With one sensor dead all day, detection recall stays at 0.90 but exact-pipe accuracy drops
to 0.59 – we show this in the evaluation tab.

**10. Why did the model flag a leak on a high-demand day?**
Unexpected demand looks like a leak: more water flows in and pressures drop. On days with
25–40 % extra demand we see 35 % false alarms. Fixes: better demand forecasts, night-flow
analysis, or feeding the forecast error to the model.

**11. How do you estimate water loss?**
Water balance: extra inflow at the reservoir meter compared with the twin's forecast. In
tests the error was about 0.05 L/s on average. If that meter fails, the estimate is shown as
unavailable instead of guessed.

**12. Why those priority weights?**
They are design choices: water loss matters most, then service impact, then certainty. They
are visible in the app and in one config file, so a utility can change them.

**13. How would you scale to a real city network?**
Localise to zones first (district metered areas), then pipes within the zone; use WNTR to
generate training data from the utility's own model; place sensors with an optimisation
method; run models per zone.

**14. Can it find two leaks at once?**
Not yet – each simulated day has one leak. Multi-label classification or iterative
“find, subtract, repeat” is on our roadmap.

**15. What would you do with one more month?**
WNTR + a public benchmark (L-Town/BattLeDIM), probability calibration, sensor placement
optimisation, and a pilot with a utility on historical data.

**16. Who in the team did what?**
(Fill in.) Member 1 models and evaluation; Member 2 simulation; Member 3 dashboard;
Member 4 integration, priority, savings, tests, docs.
