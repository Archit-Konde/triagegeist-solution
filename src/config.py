"""
Triagegeist configuration — single source of truth for paths, hyperparams, and feature lists.
"""

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
COMPLAINTS_PATH = f"{DATA_DIR}/chief_complaints.csv"
HISTORY_PATH = f"{DATA_DIR}/patient_history.csv"
SAMPLE_PATH = f"{DATA_DIR}/sample_submission.csv"

EXPERIMENTS_LOG = "experiments.jsonl"
SUBMISSION_DIR = "submissions"

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_COL = "triage_acuity"
ID_COL = "patient_id"
N_FOLDS = 5
RANDOM_STATE = 42

# ── Columns to drop ──────────────────────────────────────────────────────────
LEAKAGE_COLS = ["ed_los_hours", "disposition"]
DROP_COLS = ["triage_nurse_id", "site_id"]

# ── Vitals (for imputation) ──────────────────────────────────────────────────
VITAL_COLS = [
    "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
    "pulse_pressure", "respiratory_rate", "temperature_c",
    "shock_index", "heart_rate", "spo2", "weight_kg", "height_cm", "bmi",
]

# ── TF-IDF settings ──────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# ── LightGBM multiclass params ───────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_error",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}

# ── Glaucoma binary classifier params ────────────────────────────────────────
GLAUCOMA_PARAMS = {
    "objective": "binary",
    "metric": "binary_error",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}

GLAUCOMA_FEATURES = [
    "news2_score", "gcs_total", "pain_score", "heart_rate",
    "systolic_bp", "diastolic_bp", "respiratory_rate",
    "spo2", "temperature_c", "shock_index", "arrival_hour",
    "num_comorbidities", "num_prior_ed_visits_12m",
]
