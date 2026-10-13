"""Fitted log-sum benchmark using training data only."""
import numpy as np
from sklearn.linear_model import LinearRegression


def fit_predict(X_train, y_train, X_test):
    Xtr, Xte = np.asarray(X_train, float), np.asarray(X_test, float)
    ytr = np.asarray(y_train, float).reshape(-1)
    if Xtr.ndim != 2 or Xte.ndim != 2 or Xtr.shape[1] < 5 or Xte.shape[1] < 5 or len(Xtr) != len(ytr):
        raise ValueError("Expected aligned ages 0--4 input counts and training targets")
    if not np.isfinite(Xtr[:, :5]).all() or not np.isfinite(Xte[:, :5]).all() or not np.isfinite(ytr).all() or (Xtr[:, :5] < 0).any() or (Xte[:, :5] < 0).any() or (ytr < 0).any():
        raise ValueError("Counts and targets must be finite and non-negative")
    xtr = np.log1p(Xtr[:, :5].sum(axis=1)).reshape(-1, 1)
    xte = np.log1p(Xte[:, :5].sum(axis=1)).reshape(-1, 1)
    model = LinearRegression().fit(xtr, np.log1p(ytr))
    return np.maximum(np.expm1(model.predict(xte)), 0.0), model
