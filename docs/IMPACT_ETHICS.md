# AquaSentinel – Impact and Ethics Note

## Impact
**Water.** Leaks waste treated water that has already cost energy, chemicals and money to
produce. In our simulation a medium leak of 2 L/s loses about 170 m³ per day – roughly
5 000 m³ per month. Finding it in hours instead of weeks is where the value lies.
*(Simulated estimate, not a measured saving.)*

**Operations.** Instead of searching a whole district, crews get a ranked shortlist: the most
likely pipe, plus the next two candidates. In our tests the prediction was the leaking pipe
or a directly connected one on 98 % of detected leak days (simulated).

**Climate and resilience.** Less lost water means less pumping energy and less pressure on
water resources during dry periods – relevant for island states such as Mauritius.

**Fairness of service.** The importance score rewards pipes that serve critical facilities
(hospital, school) and carry more water, so repairs that affect more people come first.

## Ethics and responsible use
1. **Synthetic data only.** No real utility, customer or personal data is used. Every screen
   is labelled “simulated”.
2. **No overclaiming.** Metrics come from a simplified hydraulic model. They are not evidence
   of real-world performance. Real deployment requires real network data, sensor
   validation, recalibration and testing with water engineers.
3. **Human in the loop.** AquaSentinel recommends; engineers decide. It does not operate
   valves or pumps.
4. **Transparent scores.** The priority weights (0.5 / 0.3 / 0.2) are design choices shown in
   the app, not hidden model outputs. Utilities should set them with stakeholders and check
   that low-income or remote areas are not systematically placed at the bottom of the queue.
5. **Known failure modes are disclosed.** Small leaks are often missed, a dead sensor reduces
   accuracy, and unusual demand (heat waves, events) can cause false alarms. False alarms cost
   crew time; missed leaks cost water – both are reported.
6. **Confidence is not certainty.** Model scores are vote shares, not calibrated probabilities.
7. **Privacy by design for the future.** A real system would use network-level sensors, not
   household smart-meter data; if household data were ever added it would need consent,
   aggregation and data-protection compliance (e.g. Mauritius Data Protection Act 2017).
8. **Security.** A real deployment would treat sensor data and network maps as critical
   infrastructure information and restrict access.
