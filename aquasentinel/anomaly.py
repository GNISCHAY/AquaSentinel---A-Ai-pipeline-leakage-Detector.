"""
Anomaly detection with Isolation Forest (Member 1 - AI/ML Developer).

Idea: Isolation Forest builds many random trees. Unusual points are easy to
"isolate" (they end up alone after few random splits), so they get a high
anomaly score. We train it ONLY on leak-free days, so it learns what normal
residuals look like - no leak labels are needed for detection.

Isolation Forest is good at unusual COMBINATIONS of sensors, but it has a known
blind spot: a point that is extreme on only ONE of nine sensors is isolated
only when a tree happens to split on that sensor, so its score stays modest.
(Example: a leak on the trunk main raises inlet flow by 15 standard deviations
but barely moves any pressure.) We therefore add a second, learned check:

  * range check: the largest |z-score| across sensors, compared with what was
    seen on normal days

Both checks get a threshold learned from SEPARATE leak-free calibration days
(99.5% quantile; the range check also gets a 1.5x safety margin so that noisy
sensors do not flood the crew with false alarms). They are combined into one
score where 1.0 = threshold:

    anomaly score = max(IF score / IF threshold, max|z| / z threshold)

To reduce false alarms an ALARM needs 3 consecutive anomalous steps (45 min),
so a single noisy reading does not trigger a crew call-out.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from aquasentinel import config


class AnomalyDetector:
    def __init__(self, feature_names, seed=config.GLOBAL_SEED):
        self.feature_names = list(feature_names)
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
        self.if_threshold = None
        self.z_threshold = None
        self.threshold = 1.0          # combined score threshold (by construction)
        self.persistence = config.ALARM_PERSISTENCE_STEPS

    def fit(self, normal_features, calibration_features=None):
        """
        Train on features from leak-free scenarios only.

        The thresholds are set on a SEPARATE set of leak-free days
        (calibration_features) so they are not tuned on the training data itself.
        """
        values = normal_features[self.feature_names].to_numpy()
        scaled = self.scaler.fit_transform(values)
        self.model.fit(scaled)
        if calibration_features is None:
            calibration_features = normal_features
        quantile = 1 - config.NORMAL_SCORE_QUANTILE
        self.if_threshold = float(np.quantile(self.isolation_scores(calibration_features), quantile))
        self.z_threshold = config.RANGE_CHECK_MARGIN * float(
            np.quantile(self.max_abs_z(calibration_features), quantile))
        return self

    def z_scores(self, features):
        """How many 'normal standard deviations' each reading is away from normal."""
        return self.scaler.transform(features[self.feature_names].to_numpy())

    def isolation_scores(self, features):
        """Raw Isolation Forest score per step (higher = more unusual)."""
        # sklearn returns "higher = more normal"; we flip it so higher = more anomalous
        return -self.model.decision_function(self.z_scores(features))

    def max_abs_z(self, features):
        """Largest absolute z-score across all sensors, per step."""
        return np.abs(self.z_scores(features)).max(axis=1)

    def anomaly_scores(self, features):
        """Combined score per step: 1.0 = alarm threshold, higher = more unusual."""
        return np.maximum(self.isolation_scores(features) / self.if_threshold,
                          self.max_abs_z(features) / self.z_threshold)

    def detect(self, features):
        """
        Run detection over one scenario (time-ordered features).

        Returns dict:
            scores      combined anomaly score per step (1.0 = threshold)
            if_scores   Isolation Forest part, scaled the same way
            range_scores  range-check part, scaled the same way
            anomalous   bool per step (score above threshold)
            alarm       bool per step (anomalous for `persistence` steps in a row)
            first_alarm_step  index of the first alarm, or None
        """
        if_part = self.isolation_scores(features) / self.if_threshold
        range_part = self.max_abs_z(features) / self.z_threshold
        scores = np.maximum(if_part, range_part)
        anomalous = scores > self.threshold
        alarm = np.zeros_like(anomalous)
        run_length = 0
        for t, is_anomalous in enumerate(anomalous):
            run_length = run_length + 1 if is_anomalous else 0
            alarm[t] = run_length >= self.persistence
        alarm_steps = np.flatnonzero(alarm)
        first_alarm = int(alarm_steps[0]) if len(alarm_steps) else None
        return {"scores": scores, "if_scores": if_part, "range_scores": range_part,
                "anomalous": anomalous, "alarm": alarm, "first_alarm_step": first_alarm}
