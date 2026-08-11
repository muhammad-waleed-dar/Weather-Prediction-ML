"""
Data preprocessing — cleaning only. No feature engineering, no encoding.

Reproduces Phase 1's cleaning steps so training and any future re-cleaning
work reuse the exact same logic instead of two copies drifting apart.
"""

import pandas as pd


def clean_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the raw weatherAUS.csv and applies Phase 1 cleaning:
      - drop duplicate rows
      - drop rows with a missing target (RainTomorrow)
      - impute nulls: mode for categorical columns, median for numeric
      - remove Rainfall outliers via IQR (carried over from Phase 1/3)

    Returns a cleaned DataFrame with 'Date' still intact — feature
    engineering (Month/Season/etc.) happens separately in
    feature_engineering.py.
    """
    df = pd.read_csv(csv_path)

    df = df.drop_duplicates()
    df.dropna(subset=["RainTomorrow"], inplace=True)

    for col in df.columns:
        # Use is_numeric_dtype rather than checking dtype == "object" directly —
        # pandas 3.0+ uses a dedicated "str" dtype for text columns instead of
        # "object", which the old check silently missed (columns like Location
        # or WindGustDir would fall through to .median() and crash).
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    Q1 = df["Rainfall"].quantile(0.25)
    Q3 = df["Rainfall"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df = df[(df["Rainfall"] >= lower) & (df["Rainfall"] <= upper)]

    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})
    return df
