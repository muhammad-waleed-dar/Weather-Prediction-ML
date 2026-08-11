"""
Prediction pipeline — turns one dict of raw feature values into a
Rain / No Rain prediction. Uses model_loader for the artifacts and
feature_engineering for encoding, so this file only contains the
inference sequence itself.
"""

import pandas as pd

from model_loader import ModelBundle
from feature_engineering import encode_categoricals
from utils import CATEGORICAL_COLUMNS, DECISION_THRESHOLD


class RainPredictor:
    def __init__(self):
        self.bundle = ModelBundle()

    def categorical_options(self, column: str):
        return self.bundle.categorical_options(column)

    def predict(self, raw_input: dict, threshold: float = DECISION_THRESHOLD):
        """
        raw_input: dict with the 26 raw (pre-encoding, pre-scaling) feature
        values, keyed by the same names as utils.FEATURE_COLUMNS.

        Returns: (label: "Rain"/"No Rain", probability_of_rain: float)
        """
        df = pd.DataFrame([raw_input])

        # Guarantee exact training column order before encoding/scaling
        df = df[self.bundle.feature_columns]

        # Encode categoricals with the SAME fitted encoders from training
        df, _ = encode_categoricals(df, CATEGORICAL_COLUMNS, encoders=self.bundle.encoders)

        # Scale with the SAME fitted scaler from training
        df_scaled = pd.DataFrame(
            self.bundle.scaler.transform(df), columns=self.bundle.feature_columns
        )

        prob_rain = self.bundle.model.predict_proba(df_scaled)[0, 1]
        label = "Rain" if prob_rain >= threshold else "No Rain"
        return label, float(prob_rain)
