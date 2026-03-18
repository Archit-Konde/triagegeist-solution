#!/usr/bin/env python
"""Triagegeist pipeline — LightGBM multiclass baseline."""
import numpy as np
import lightgbm as lgb
from features import engineer_features, apply_features

PARAMS = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_error",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def train_and_predict(X_train, y_train, X_val, y_val):
    # Shift labels from 1-5 to 0-4 for LightGBM
    y_train_shifted = y_train - 1
    y_val_shifted = y_val - 1

    X_train_fe, fit_params = engineer_features(X_train.copy(), is_train=True)
    X_val_fe = apply_features(X_val.copy(), fit_params)

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        X_train_fe, y_train_shifted,
        eval_set=[(X_val_fe, y_val_shifted)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    preds_shifted = model.predict(X_val_fe)
    return preds_shifted + 1  # shift back to 1-5
