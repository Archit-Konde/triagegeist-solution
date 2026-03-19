"""Triagegeist pipeline — LightGBM multiclass baseline."""
import lightgbm as lgb
from features import engineer_features, apply_features
from config import LGBM_PARAMS


def train_and_predict(X_train, y_train, X_val, y_val):
    assert y_train.min() >= 1 and y_train.max() <= 5, f"Labels out of range: [{y_train.min()}, {y_train.max()}]"

    # Shift labels from 1-5 to 0-4 for LightGBM
    y_train_shifted = y_train - 1
    y_val_shifted = y_val - 1

    X_train_fe, fit_params = engineer_features(X_train.copy(), is_train=True)
    X_val_fe = apply_features(X_val.copy(), fit_params)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
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
