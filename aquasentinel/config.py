"""
Central configuration for AquaSentinel.

Every "magic number" used by the project lives here so that the team can
tune the system in one place. All values describe a SYNTHETIC network.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"
LOCALIZER_MODEL_PATH = MODELS_DIR / "leak_localizer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

# ---------------------------------------------------------------------------
# Time settings: one scenario = one simulated day
# ---------------------------------------------------------------------------
TIMESTEP_MINUTES = 15
STEPS_PER_DAY = 24 * 60 // TIMESTEP_MINUTES  # 96 steps

# ---------------------------------------------------------------------------
# Hydraulic settings (simplified solver, see hydraulics.py)
# ---------------------------------------------------------------------------
RESERVOIR_HEAD_M = 75.0          # total head at the reservoir (metres)
HAZEN_WILLIAMS_C = 110.0         # pipe roughness coefficient (older pipes)
SOLVER_ITERATIONS = 40           # fixed-point iterations of the solver

# Leak model: leak_flow (L/s) = coefficient * sqrt(pressure in metres)
# Severity in the UI is a number from 0 to 1 that maps onto this range.
LEAK_COEFFICIENT_MIN = 0.03      # severity 0  -> ~0.2 L/s at 40 m pressure
LEAK_COEFFICIENT_MAX = 0.45      # severity 1  -> ~2.8 L/s at 40 m pressure
SEVERITY_PRESETS = {"small": 0.15, "medium": 0.45, "large": 0.85}

# ---------------------------------------------------------------------------
# Sensor noise defaults (noise_level = 1.0 means "typical" noise)
# ---------------------------------------------------------------------------
PRESSURE_NOISE_STD_M = 0.15      # metres of head
FLOW_NOISE_FRACTION = 0.01       # 1% of the reading
NODE_DEMAND_VARIATION = 0.08     # +/- random variation of each customer node
GLOBAL_DEMAND_VARIATION = 0.02   # unexpected network-wide demand drift

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
ROLLING_WINDOW_STEPS = 4         # 4 x 15 min = 1 hour smoothing window

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
ALARM_PERSISTENCE_STEPS = 3      # anomaly must persist 3 steps (45 min)
NORMAL_SCORE_QUANTILE = 0.005    # threshold = 0.5% quantile of normal scores
# Safety margin on the single-sensor range check. 1.0 caught more leaks but gave
# ~80% false alarms when sensor noise tripled; 1.5 kept them near 4%. Chosen on
# separate tuning seeds (3001-3005), NOT on the reported test sets.
RANGE_CHECK_MARGIN = 1.5

# ---------------------------------------------------------------------------
# Repair prioritisation (DESIGN CHOICES - not derived from data)
# ---------------------------------------------------------------------------
PRIORITY_WEIGHTS = {"water_loss": 0.5, "importance": 0.3, "confidence": 0.2}
REFERENCE_LOSS_LPS = 3.0         # a loss of 3 L/s or more scores 1.0

# ---------------------------------------------------------------------------
# Savings estimation (ASSUMPTION - replace with a real tariff)
# ---------------------------------------------------------------------------
DEFAULT_COST_PER_M3 = 20.0
DEFAULT_CURRENCY_LABEL = "cost units"
DAYS_PER_MONTH = 30
