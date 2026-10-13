"""Shared XGBoost fitting helper used by the six family-specific modules."""
import numpy as np
from xgboost import XGBRegressor


def fit_predict_counts(X_train, y_train, X_test, params, random_state=42):
    """Fit on log1p(counts) and return non-negative count-scale predictions."""
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test matrices must have the same feature columns")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train are not aligned")
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         random_state=random_state, **params)
    model.fit(X_train, np.log1p(y_train))
    return np.maximum(np.expm1(model.predict(X_test)), 0.0), model
