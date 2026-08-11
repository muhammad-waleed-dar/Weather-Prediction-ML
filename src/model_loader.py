"""
Model loading — loads the serialized model, scaler, encoders, and feature
column order from disk. Kept separate from predict.py so "load the model"
and "use the model to predict" are two independently testable steps.
"""

import os
import joblib

from utils import MODEL_PATH, SCALER_PATH, ENCODERS_PATH, FEATURE_COLUMNS_PATH


class ModelBundle:
    """Holds every artifact produced by train_model.py, loaded once."""

    def __init__(self):
        for path, label in [
            (MODEL_PATH, "model"), (SCALER_PATH, "scaler"),
            (ENCODERS_PATH, "encoders"), (FEATURE_COLUMNS_PATH, "feature columns"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {label} file at {path}. Run `python src/train_model.py` "
                    f"first to generate it."
                )

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.encoders = joblib.load(ENCODERS_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    def categorical_options(self, column: str):
        """Category values the encoder actually saw during training —
        use this to populate UI dropdowns so an unseen value can never
        be submitted."""
        return list(self.encoders[column].classes_)
