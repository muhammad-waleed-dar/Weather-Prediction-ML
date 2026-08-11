"""
Main training script — orchestrates preprocessing, feature engineering,
and model training, then serializes everything the app needs.

Run from the project root:
    python src/train_model.py

Requires weatherAUS.csv to be placed in data/weatherAUS.csv.
"""

import os
import joblib
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from imblearn.combine import SMOTETomek

from preprocessing import clean_raw_data
from feature_engineering import (
    engineer_date_features, engineer_numeric_features, encode_categoricals,
)
from utils import (
    DATA_PATH, MODELS_DIR, MODEL_PATH, SCALER_PATH,
    ENCODERS_PATH, FEATURE_COLUMNS_PATH, CATEGORICAL_COLUMNS,
    DECISION_THRESHOLD,
)


def main():
    print(f"Loading and cleaning data from {DATA_PATH} ...")
    df = clean_raw_data(DATA_PATH)
    print(f"Shape after cleaning: {df.shape}")

    df = engineer_date_features(df)
    df = engineer_numeric_features(df)
    print(f"Shape after feature engineering: {df.shape}")

    df, encoders = encode_categoricals(df, CATEGORICAL_COLUMNS)  # fit mode

    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]
    feature_columns = list(X.columns)  # exact order the scaler/model expect

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    print("Random Forest trained.")

    probs = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_default = model.predict(X_test_scaled)
    y_pred_tuned = (probs >= DECISION_THRESHOLD).astype(int)

    print(f"Default (0.5)              — Recall: {recall_score(y_test, y_pred_default):.4f}, "
          f"Precision: {precision_score(y_test, y_pred_default):.4f}, "
          f"F1: {f1_score(y_test, y_pred_default):.4f}")
    print(f"Tuned ({DECISION_THRESHOLD})               — Recall: {recall_score(y_test, y_pred_tuned):.4f}, "
          f"Precision: {precision_score(y_test, y_pred_tuned):.4f}, "
          f"F1: {f1_score(y_test, y_pred_tuned):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    print(f"\nSaved to {MODELS_DIR}/:")
    print("  - random_forest_model.joblib")
    print("  - scaler.joblib")
    print("  - encoders.joblib")
    print("  - feature_columns.joblib")


if __name__ == "__main__":
    main()
