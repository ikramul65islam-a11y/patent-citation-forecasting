"""XGBoost model using citation descriptors and frozen patent-text vectors."""
import numpy as np
from .common import fit_predict_counts


def build_features(citation_21, text_vectors):
    c, t = np.asarray(citation_21, np.float32), np.asarray(text_vectors, np.float32)
    if c.ndim != 2 or c.shape[1] != 21 or t.ndim != 2 or t.shape[1] != 384 or len(c) != len(t):
        raise ValueError("Expected aligned citation (21) and text (384) columns")
    return np.concatenate((c, t), axis=1)


def fit_predict(citation_train, text_train, y_train, citation_test, text_test, params, random_state=42):
    return fit_predict_counts(build_features(citation_train, text_train), y_train,
                             build_features(citation_test, text_test), params, random_state)
