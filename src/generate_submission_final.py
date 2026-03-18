#!/usr/bin/env python
"""
Final submission — 3-tier + glaucoma-specific binary classifier.

Tier 1:  Unambiguous complaint text → direct lookup (19885 rows)
Tier 2:  Glaucoma-specific LightGBM binary classifier (76 rows)
Tier 3:  Full LightGBM multiclass model (57 unseen complaint texts)
"""
import pandas as pd, numpy as np, lightgbm as lgb
from features import engineer_features, apply_features

PARAMS = {
    "objective": "multiclass", "num_class": 5, "metric": "multi_error",
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "class_weight": "balanced",
    "random_state": 42, "verbose": -1, "n_jobs": -1,
}
GLAUCOMA_PARAMS = {
    "objective": "binary", "metric": "binary_error",
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 15,
    "random_state": 42, "verbose": -1, "n_jobs": -1,
}
GLAUCOMA_FEATURES = ["news2_score", "gcs_total", "pain_score", "heart_rate",
                     "systolic_bp", "diastolic_bp", "respiratory_rate",
                     "spo2", "temperature_c", "shock_index", "arrival_hour",
                     "num_comorbidities", "num_prior_ed_visits_12m"]


def main():
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    complaints = pd.read_csv("data/chief_complaints.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    TARGET = "triage_acuity"

    train_c = train.merge(complaints[["patient_id","chief_complaint_raw"]], on="patient_id", how="left")
    test_c  = test.merge(complaints[["patient_id","chief_complaint_raw"]],  on="patient_id", how="left")

    # ── Tier 1: unambiguous complaint lookup ────────────────────────────────
    per_text = train_c.groupby("chief_complaint_raw")[TARGET].nunique()
    unamb = per_text[per_text == 1].index
    amb   = per_text[per_text > 1].index
    tier1 = train_c[train_c["chief_complaint_raw"].isin(unamb)].groupby("chief_complaint_raw")[TARGET].first().to_dict()

    preds = np.full(len(test), -1, dtype=int)
    sources = ["unresolved"] * len(test)

    for i, row in test_c.iterrows():
        idx = test_c.index.get_loc(i)
        if row["chief_complaint_raw"] in tier1:
            preds[idx] = tier1[row["chief_complaint_raw"]]
            sources[idx] = "tier1"

    print(f"Tier 1 resolved: {sources.count('tier1')}")

    # ── Tier 2: glaucoma-specific binary model (acuity 1 vs 2) ─────────────
    glaucoma_train = train_c[train_c["chief_complaint_raw"].isin(amb)].copy()
    glaucoma_test  = test_c[test_c["chief_complaint_raw"].isin(amb)].copy()
    glaucoma_test_idx = glaucoma_test.index.map(test_c.index.get_loc)

    # Impute missing vitals with training medians
    fit_meds = {}
    for col in GLAUCOMA_FEATURES:
        fit_meds[col] = glaucoma_train[col].median()
        glaucoma_train[col] = glaucoma_train[col].fillna(fit_meds[col])
        glaucoma_test[col]  = glaucoma_test[col].fillna(fit_meds[col])

    # Binary target: 1 vs 2
    gl_y = (glaucoma_train[TARGET] == 1).astype(int).values
    gl_X = glaucoma_train[GLAUCOMA_FEATURES].values
    gl_X_test = glaucoma_test[GLAUCOMA_FEATURES].values

    print(f"\nGlaucoma-specific classifier: {len(glaucoma_train)} train, {len(glaucoma_test)} test rows")
    gl_model = lgb.LGBMClassifier(**GLAUCOMA_PARAMS)
    gl_model.fit(gl_X, gl_y, callbacks=[lgb.log_evaluation(0)])
    gl_preds_bin = gl_model.predict(gl_X_test)
    gl_preds = np.where(gl_preds_bin == 1, 1, 2)

    for j, i in enumerate(glaucoma_test.index):
        idx = test_c.index.get_loc(i)
        preds[idx] = gl_preds[j]
        sources[idx] = "tier2_glaucoma"

    print(f"Tier 2 resolved: {sources.count('tier2_glaucoma')}")

    # ── Tier 3: full model for unseen complaint texts ───────────────────────
    model_needed = [i for i, s in enumerate(sources) if s == "unresolved"]
    print(f"\nTier 3 model fallback: {len(model_needed)} rows")

    if model_needed:
        X_train = train.drop(columns=[TARGET])
        y_train = train[TARGET].values - 1
        X_train_fe, fit_params = engineer_features(X_train.copy(), is_train=True)
        X_test_fe = apply_features(test.copy(), fit_params)
        full_model = lgb.LGBMClassifier(**PARAMS)
        full_model.fit(X_train_fe, y_train, callbacks=[lgb.log_evaluation(0)])
        model_raw = full_model.predict(X_test_fe) + 1
        for idx in model_needed:
            preds[idx] = model_raw[idx]
            sources[idx] = "tier3_model"

    assert (preds != -1).all(), "Some rows unresolved!"
    assert (preds >= 1).all() and (preds <= 5).all()

    submission = pd.DataFrame({"patient_id": test["patient_id"], "triage_acuity": preds.astype(int)})
    assert list(submission.columns) == list(sample_sub.columns)

    out = "submissions/submission_final.csv"
    submission.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Distribution:\n{submission['triage_acuity'].value_counts().sort_index()}")
    print(f"\nSources: tier1={sources.count('tier1')}, tier2={sources.count('tier2_glaucoma')}, tier3={sources.count('tier3_model')}")


if __name__ == "__main__":
    main()
