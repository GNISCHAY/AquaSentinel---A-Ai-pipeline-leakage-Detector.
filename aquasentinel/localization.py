"""
Leak localisation with a Random Forest classifier (Member 1 - AI/ML Developer).

Idea: a leak in each pipe leaves a different "fingerprint" on the sensors
(which pressures drop most, how much extra water enters the network).
We simulate many labelled leaks, then train a Random Forest to map a
fingerprint -> the pipe that caused it.

Confidence: the score we show is the fraction of the forest's trees voting
for a pipe. It is a RANKING SCORE, not a calibrated probability.

Explanation: we combine
  * how unusual each sensor is right now (z-score vs. normal days), and
  * how much the forest relies on that sensor overall (feature importance)
and we compare the observed pattern with the average learned fingerprint of
the predicted pipe (cosine similarity).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from aquasentinel import config


class LeakLocalizer:
    def __init__(self, feature_names, seed=config.GLOBAL_SEED):
        self.feature_names = list(feature_names)
        self.model = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed,
        )
        self.fingerprints = {}   # pipe -> average (unit-length) z-score pattern

    def fit(self, leak_features, labels, z_scores):
        """
        leak_features : DataFrame of features from time steps where a leak is active
        labels        : pipe name for each row
        z_scores      : the same rows standardised by the anomaly detector
                        (used only to build the explanation fingerprints)
        """
        self.model.fit(leak_features[self.feature_names].to_numpy(), np.asarray(labels))
        # Train on all CPU cores, but predict single-threaded: predictions are tiny,
        # and parallel prediction inside Streamlit only adds overhead and warnings.
        self.model.set_params(n_jobs=1)
        labels = np.asarray(labels)
        for pipe in np.unique(labels):
            mean_pattern = z_scores[labels == pipe].mean(axis=0)
            self.fingerprints[pipe] = mean_pattern / (np.linalg.norm(mean_pattern) + 1e-9)
        return self

    @property
    def classes(self):
        return list(self.model.classes_)

    def predict_window(self, window_features):
        """
        Predict over a window of time steps (e.g. every step since the alarm).
        Per-step tree votes are averaged, which is more stable than one step.
        Returns a DataFrame [pipe, score] sorted from most to least likely.
        """
        votes = self.model.predict_proba(window_features[self.feature_names].to_numpy())
        mean_votes = votes.mean(axis=0)
        ranking = pd.DataFrame({"pipe": self.classes, "score": mean_votes})
        return ranking.sort_values("score", ascending=False).reset_index(drop=True)

    def explain(self, window_z_scores, predicted_pipe):
        """
        Explain a prediction.
        window_z_scores : array (steps, sensors) of z-scores for the window
        Returns (DataFrame of per-sensor contributions, fingerprint similarity).
        """
        observed = window_z_scores.mean(axis=0)
        importance = self.model.feature_importances_
        contribution = importance * np.abs(observed)
        share = contribution / (contribution.sum() + 1e-9)
        table = pd.DataFrame({
            "sensor": self.feature_names,
            "deviation_z": np.round(observed, 2),
            "global_importance": np.round(importance, 3),
            "contribution_share": np.round(share, 3),
        }).sort_values("contribution_share", ascending=False).reset_index(drop=True)

        fingerprint = self.fingerprints.get(predicted_pipe)
        similarity = None
        if fingerprint is not None:
            unit_observed = observed / (np.linalg.norm(observed) + 1e-9)
            similarity = float(np.dot(unit_observed, fingerprint))
        return table, similarity
