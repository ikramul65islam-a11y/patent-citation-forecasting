"""XGBoost model using citation descriptors and financial-report activity."""
import numpy as np
from .common import fit_predict_counts


def build_features(citation_21, finance_21, source_years_covered):
    c, f = np.asarray(citation_21, np.float32), np.asarray(finance_21, np.float32)
    coverage = np.asarray(source_years_covered, np.float32).reshape(-1, 1)
    if c.ndim != 2 or c.shape[1] != 21 or f.ndim != 2 or f.shape[1] != 21 or len(c) != len(f) or len(c) != len(coverage):
        raise ValueError("Expected aligned citation (21), finance (21), and coverage columns")
    return np.concatenate((c, f, coverage), axis=1)


def fit_predict(citation_train, finance_train, coverage_train, y_train,
                citation_test, finance_test, coverage_test, params, random_state=42):
    return fit_predict_counts(build_features(citation_train, finance_train, coverage_train), y_train,
                             build_features(citation_test, finance_test, coverage_test), params, random_state)
