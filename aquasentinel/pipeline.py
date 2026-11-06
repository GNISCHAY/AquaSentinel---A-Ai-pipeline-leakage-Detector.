"""
Integration pipeline (Member 4 - Integration & Product Developer).

This is the ONE place where all modules meet. The dashboard and the
evaluation both call `analyse_scenario`, so what judges see in the demo is
exactly what was evaluated.

    simulation -> features -> anomaly detection -> localisation
               -> water loss estimate -> savings -> repair priority
"""

import joblib

from aquasentinel import config
from aquasentinel.features import INLET_FLOW_FEATURE, compute_residuals, scenario_features
from aquasentinel.prioritization import priority_band, priority_score
from aquasentinel.savings import estimate_savings

SUSPICIOUS_SCORE = 0.10      # pipes with at least 10% of tree votes are "suspicious"
MAX_SUSPICIOUS = 3


def save_models(detector, localizer, pipe_importance):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, config.ANOMALY_MODEL_PATH)
    joblib.dump({"localizer": localizer, "pipe_importance": pipe_importance},
                config.LOCALIZER_MODEL_PATH)


def models_exist():
    return config.ANOMALY_MODEL_PATH.exists() and config.LOCALIZER_MODEL_PATH.exists()


def load_models():
    """Returns (detector, localizer, pipe_importance)."""
    detector = joblib.load(config.ANOMALY_MODEL_PATH)
    bundle = joblib.load(config.LOCALIZER_MODEL_PATH)
    return detector, bundle["localizer"], bundle["pipe_importance"]


def estimate_leak_flow(simulation, start_step):
    """
    Estimated leak = average extra inflow at the reservoir meter since the alarm
    (measured inflow - twin's expected inflow). Returns (L/s, is_available).
    """
    raw = simulation["measured"][INLET_FLOW_FEATURE].iloc[start_step:]
    if raw.isna().all():
        return 0.0, False
    residual = compute_residuals(simulation["measured"], simulation["expected"])
    valid = raw.notna()
    extra = residual[INLET_FLOW_FEATURE].iloc[start_step:][valid]
    return max(float(extra.mean()), 0.0), True


def analyse_scenario(simulation, detector, localizer, pipe_importance,
                     cost_per_m3=config.DEFAULT_COST_PER_M3,
                     currency=config.DEFAULT_CURRENCY_LABEL):
    """
    Run the full AI pipeline on one simulated day. Uses ONLY sensor data and the
    digital twin - the hidden ground truth is never passed to the models.
    """
    features = scenario_features(simulation)
    detection = detector.detect(features)
    first_alarm = detection["first_alarm_step"]

    result = {
        "features": features,
        "detection": detection,
        "status": "NORMAL",
        "ranking": None,
        "predicted_pipe": None,
        "confidence": 0.0,
        "suspicious_pipes": [],
        "explanation": None,
        "fingerprint_similarity": None,
        "leak_flow_lps": 0.0,
        "inlet_meter_available": True,
        "savings": None,
        "priority": None,
        "priority_parts": None,
        "action": None,
    }

    if first_alarm is None:
        # Some isolated anomalous readings but no persistent alarm -> "watch"
        if detection["anomalous"].any():
            result["status"] = "WATCH"
        return result

    # --- Localise using every time step since the alarm ---------------------
    window = features.iloc[first_alarm:]
    ranking = localizer.predict_window(window)
    predicted_pipe = ranking.loc[0, "pipe"]
    confidence = float(ranking.loc[0, "score"])
    suspicious = ranking[ranking["score"] >= SUSPICIOUS_SCORE]["pipe"].head(MAX_SUSPICIOUS).tolist()
    explanation, similarity = localizer.explain(detector.z_scores(window), predicted_pipe)

    # --- Impact ---------------------------------------------------------------
    leak_flow, inlet_ok = estimate_leak_flow(simulation, first_alarm)
    importance = pipe_importance.get(predicted_pipe, 0.0)
    priority, parts = priority_score(leak_flow, importance, confidence)

    result.update({
        "status": "LEAK SUSPECTED",
        "ranking": ranking,
        "predicted_pipe": predicted_pipe,
        "confidence": confidence,
        "suspicious_pipes": suspicious,
        "explanation": explanation,
        "fingerprint_similarity": similarity,
        "leak_flow_lps": leak_flow,
        "inlet_meter_available": inlet_ok,
        "savings": estimate_savings(leak_flow, cost_per_m3, currency),
        "priority": priority,
        "priority_parts": parts,
        "action": priority_band(priority),
    })
    return result
