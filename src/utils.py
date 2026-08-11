"""
Shared constants and paths used across the training and inference scripts.
Keeping these in one place avoids the feature-order / column-name mismatches
that break MinMaxScaler and LabelEncoder at inference time.
"""

import os

# ------------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "weatherAUS.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")

# ------------------------------------------------------------------------------
# FEATURE SCHEMA — must exactly match the column order X was built with in
# Phase 3 (ModelTraining_Evaluation.ipynb, Step 2/3). Do not reorder.
# ------------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
    "RainToday", "Month", "Season", "TempRange", "HumidityChange", "PressureChange",
]

# Columns that were Label-Encoded in Phase 3 Step 2 — same 7 columns here.
CATEGORICAL_COLUMNS = [
    "Location", "WindGustDir", "WindDir9am", "WindDir3pm",
    "RainToday", "Month", "Season",
]

# Chosen operating threshold from Phase 3 threshold-tuning (README:
# Rain Recall 55.95% -> 76.47% at Precision 43.47%, vs default 0.5).
DECISION_THRESHOLD = 0.35


def month_to_season(month_num: int) -> str:
    """Same mapping used in Phase 1/2/3 notebooks — kept here so inference
    can derive Season from a selected month without duplicating the logic."""
    if month_num in (12, 1, 2):
        return "Summer"
    elif month_num in (3, 4, 5):
        return "Autumn"
    elif month_num in (6, 7, 8):
        return "Winter"
    else:
        return "Spring"
