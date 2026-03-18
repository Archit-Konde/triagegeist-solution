#!/usr/bin/env python
"""
Locked evaluation pipeline — DO NOT MODIFY.
Stratified 5-fold CV. Calls train_and_predict() from pipeline.py.
"""
import json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from pipeline import train_and_predict

TRAIN_PATH = "data/train.csv"
TARGET_COL = "triage_acuity"
ID_COL = "patient_id"
N_FOLDS = 5
RANDOM_STATE = 42

def main():
    train = pd.read_csv(TRAIN_PATH)
    X = train.drop(columns=[TARGET_COL])  # keep patient_id so features.py can merge aux tables
    y = train[TARGET_COL].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        preds = train_and_predict(X_train, y_train, X_val, y_val)
        score = accuracy_score(y_val, preds)
        fold_scores.append(score)
        print(f"Fold {fold}: {score:.4f}")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"\nCV Score: {mean_score:.4f} (+/- {std_score:.4f})")

    # Append to experiments log
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cv_score": round(mean_score, 6),
        "cv_std": round(std_score, 6),
        "fold_scores": [round(s, 6) for s in fold_scores],
        "type": "cv_run",
        "status": "pending_keep_or_revert"
    }
    with open("experiments.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
