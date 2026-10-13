"""XGBoost model using citation descriptors and 1,024 cluster-centroid values."""
import numpy as np
from .common import fit_predict_counts


def build_features(citation_21, centroid_1024):
    c, z = np.asarray(citation_21, np.float32), np.asarray(centroid_1024, np.float32)
    if c.ndim != 2 or c.shape[1] != 21 or z.ndim != 2 or z.shape[1] != 1024 or len(c) != len(z):
        raise ValueError("Expected aligned citation (21) and centroid (1,024) columns")
    return np.concatenate((c, z), axis=1)


def fit_predict(citation_train, centroid_train, y_train, citation_test, centroid_test, params, random_state=42):
    return fit_predict_counts(build_features(citation_train, centroid_train), y_train,
                             build_features(citation_test, centroid_test), params, random_state)
