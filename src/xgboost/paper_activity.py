"""XGBoost model using citation descriptors and two paper-activity series."""
import numpy as np
from .common import fit_predict_counts


def build_features(citation_21, paper_42, source_years_covered):
    c, p = np.asarray(citation_21, np.float32), np.asarray(paper_42, np.float32)
    coverage = np.asarray(source_years_covered, np.float32).reshape(-1, 1)
    if c.ndim != 2 or c.shape[1] != 21 or p.ndim != 2 or p.shape[1] != 42 or len(c) != len(p) or len(c) != len(coverage):
        raise ValueError("Expected aligned citation (21), paper (42), and coverage columns")
    return np.concatenate((c, p, coverage), axis=1)


def fit_predict(citation_train, paper_train, coverage_train, y_train,
                citation_test, paper_test, coverage_test, params, random_state=42):
    return fit_predict_counts(build_features(citation_train, paper_train, coverage_train), y_train,
                             build_features(citation_test, paper_test, coverage_test), params, random_state)
