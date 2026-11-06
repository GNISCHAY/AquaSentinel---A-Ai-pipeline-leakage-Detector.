"""
Repair prioritisation (Member 4 - Integration & Product Developer).

Priority = 0.5 x Water Loss Score
         + 0.3 x Network Importance Score
         + 0.2 x Confidence Score

Every score is first scaled to 0..1 so the weights are comparable:
  * Water Loss Score  = estimated leak flow / 3 L/s  (capped at 1)
  * Importance Score  = pipe importance from network.compute_pipe_importance
  * Confidence Score  = Random Forest vote share for that pipe

The weights are DESIGN CHOICES (water loss first, then service impact,
then how sure the AI is). A utility would tune them with its engineers.
"""

import pandas as pd

from aquasentinel import config


def water_loss_score(leak_flow_lps):
    return min(max(leak_flow_lps, 0.0) / config.REFERENCE_LOSS_LPS, 1.0)


def priority_score(leak_flow_lps, importance, confidence, weights=None):
    """Return (priority, breakdown dict) for one suspected leak."""
    weights = weights or config.PRIORITY_WEIGHTS
    parts = {
        "water_loss_score": water_loss_score(leak_flow_lps),
        "importance_score": float(importance),
        "confidence_score": float(confidence),
    }
    priority = (weights["water_loss"] * parts["water_loss_score"]
                + weights["importance"] * parts["importance_score"]
                + weights["confidence"] * parts["confidence_score"])
    return round(priority, 3), parts


def priority_band(priority):
    if priority >= 0.6:
        return "CRITICAL - dispatch now"
    if priority >= 0.4:
        return "HIGH - repair within 24 h"
    if priority >= 0.25:
        return "MEDIUM - schedule this week"
    return "LOW - monitor"


def rank_incidents(incidents, pipe_importance, weights=None):
    """
    Rank a list of suspected leaks.

    incidents : list of dicts with keys
        incident_id, pipe, leak_flow_lps, confidence  (+ any extra info)
    Returns a DataFrame sorted by priority (rank 1 = repair first).
    """
    rows = []
    for incident in incidents:
        importance = pipe_importance.get(incident["pipe"], 0.0)
        priority, parts = priority_score(
            incident["leak_flow_lps"], importance, incident["confidence"], weights)
        rows.append({**incident, **parts, "importance": importance,
                     "priority": priority, "action": priority_band(priority)})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table
