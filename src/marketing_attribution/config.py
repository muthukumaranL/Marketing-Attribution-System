from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "attribution_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
JOURNEYS_PATH = ARTIFACT_DIR / "journeys.csv"
DATA_METRICS_PATH = ARTIFACT_DIR / "data_metrics.json"
ATTRIBUTION_PATH = ARTIFACT_DIR / "attribution_comparison.csv"
MARKOV_DETAILS_PATH = ARTIFACT_DIR / "markov_removal_effects.csv"
TRANSITION_MATRIX_PATH = ARTIFACT_DIR / "markov_transition_matrix.csv"
BUDGET_PATH = ARTIFACT_DIR / "budget_simulation.csv"
MODEL_METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_conversion_model.joblib"
LR_MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_pipeline.joblib"
DB_PATH = ARTIFACT_DIR / "marketing_attribution.db"

CHANNELS = ["Facebook", "Paid Search", "Instagram", "Online Video", "Online Display"]
ABSORBING_STATES = ["Conversion", "Null"]
MAX_PREFIX_TOUCHPOINTS = 5
PREDICTION_HORIZON_DAYS = 7
RANDOM_STATE = 42

ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
