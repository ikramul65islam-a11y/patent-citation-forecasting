"""XGBoost model using only the 21 early-citation descriptors."""
import numpy as np
from .common import fit_predict_counts


def build_features(citation_21):
    X = np.asarray(citation_21, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 21:
        raise ValueError("citation_21 must have 21 columns")
    return X


def fit_predict(citation_train, y_train, citation_test, params, random_state=42):
    return fit_predict_counts(build_features(citation_train), y_train,
                             build_features(citation_test), params, random_state)
