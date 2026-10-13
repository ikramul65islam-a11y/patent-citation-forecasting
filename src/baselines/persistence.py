"""Persistence benchmark: predict the future using the age-4 citation count."""
import numpy as np


def predict(X_test):
    X = np.asarray(X_test, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 5:
        raise ValueError("X_test must contain raw citation counts at ages 0--4")
    if not np.isfinite(X[:, :5]).all() or (X[:, :5] < 0).any():
        raise ValueError("Citation counts must be finite and non-negative")
    return X[:, 4].copy()
