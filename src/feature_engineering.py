"""
Feature engineering — turns cleaned raw data into the feature set the model
was trained on. Shared by training (fit new encoders) and inference
(reuse fitted encoders), so the two never drift out of sync.
"""

from typing import Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import month_to_season


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives Month (name) and Season from the Date column, then drops
    Date — its signal is now captured in Month/Season, same as Phase 2/3."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["Month"] = df["Date"].dt.month_name()
    df["Season"] = df["Date"].dt.month.apply(month_to_season)
    df.drop(columns=["Date"], inplace=True)
    return df


def engineer_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the three derived numeric features from Phase 2:
    TempRange, HumidityChange, PressureChange."""
    df = df.copy()
    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["HumidityChange"] = df["Humidity3pm"] - df["Humidity9am"]
    df["PressureChange"] = df["Pressure3pm"] - df["Pressure9am"]
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list,
    encoders: Optional[dict] = None,
):
    """
    Label-encodes the given categorical columns.

    Training mode (encoders=None): fits a new LabelEncoder per column and
    returns (encoded_df, fitted_encoders).

    Inference mode (encoders={...}): reuses the already-fitted encoders and
    validates that each value was seen during training, raising a clear
    error instead of a silent/garbage transform on an unseen category.
    """
    df = df.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    for col in categorical_cols:
        if fit_mode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            value = str(df.at[df.index[0], col]) if len(df) == 1 else None
            if value is not None and value not in le.classes_:
                raise ValueError(
                    f"'{value}' is not a recognized value for '{col}'. "
                    f"Expected one of: {list(le.classes_)}"
                )
            df[col] = le.transform(df[col].astype(str))

    return df, encoders
