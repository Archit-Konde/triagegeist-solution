"""
Final submission — 3-tier + glaucoma-specific binary classifier.

Tier 1:  Unambiguous complaint text → direct lookup (19885 rows)
Tier 2:  Glaucoma-specific LightGBM binary classifier (76 rows)
Tier 3:  Full LightGBM multiclass model (57 unseen complaint texts)
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import engineer_features, apply_features
from config import (TRAIN_PATH, TEST_PATH, COMPLAINTS_PATH, SAMPLE_PATH,
                    TARGET_COL, SUBMISSION_DIR, LGBM_PARAMS,
                    GLAUCOMA_PARAMS, GLAUCOMA_FEATURES)


def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    complaints = pd.read_csv(COMPLAINTS_PATH)
    sample_sub = pd.read_csv(SAMPLE_PATH)

    train_c = train.merge(complaints[["patient_id", "chief_complaint_raw"]], on="patient_id", how="left")
    test_c = test.merge(complaints[["patient_id", "chief_complaint_raw"]], on="patient_id", how="left")

    # ── Tier 1: unambiguous complaint lookup ────────────────────────────────
    per_text = train_c.groupby("chief_complaint_raw")[TARGET_COL].nunique()
    unamb = per_text[per_text == 1].index
    amb = per_text[per_text > 1].index
    tier1 = train_c[train_c["chief_complaint_raw"].isin(unamb)].groupby("chief_complaint_raw")[TARGET_COL].first().to_dict()

    # Vectorized lookup instead of iterrows
    preds = np.full(len(test), -1, dtype=int)
    sources = np.array(["unresolved"] * len(test), dtype=object)

    tier1_mask = test_c["chief_complaint_raw"].isin(tier1.keys())
    preds[tier1_mask.values] = test_c.loc[tier1_mask, "chief_complaint_raw"].map(tier1).values
    sources[tier1_mask.values] = "tier1"

    print(f"Tier 1 resolved: {(sources == 'tier1').sum()}")

    # ── Tier 2: glaucoma-specific binary model (acuity 1 vs 2) ─────────────
    glaucoma_train = train_c[train_c["chief_complaint_raw"].isin(amb)].copy()
    glaucoma_test = test_c[test_c["chief_complaint_raw"].isin(amb)].copy()
    glaucoma_test_positions = np.where(test_c["chief_complaint_raw"].isin(amb))[0]

    # Impute missing vitals with training medians
    fit_meds = {}
    for col in GLAUCOMA_FEATURES:
        fit_meds[col] = glaucoma_train[col].median()
        glaucoma_train[col] = glaucoma_train[col].fillna(fit_meds[col])
        glaucoma_test[col] = glaucoma_test[col].fillna(fit_meds[col])

    # Binary target: 1 vs 2
    gl_y = (glaucoma_train[TARGET_COL] == 1).astype(int).values
    gl_X = glaucoma_train[GLAUCOMA_FEATURES].values
    gl_X_test = glaucoma_test[GLAUCOMA_FEATURES].values

    print(f"\nGlaucoma-specific classifier: {len(glaucoma_train)} train, {len(glaucoma_test)} test rows")
    gl_model = lgb.LGBMClassifier(**GLAUCOMA_PARAMS)
    gl_model.fit(gl_X, gl_y, callbacks=[lgb.log_evaluation(0)])
    gl_preds_bin = gl_model.predict(gl_X_test)
    gl_preds = np.where(gl_preds_bin == 1, 1, 2)

    preds[glaucoma_test_positions] = gl_preds
    sources[glaucoma_test_positions] = "tier2_glaucoma"

    print(f"Tier 2 resolved: {(sources == 'tier2_glaucoma').sum()}")

    # ── Tier 3: full model for unseen complaint texts ───────────────────────
    model_needed = np.where(sources == "unresolved")[0]
    print(f"\nTier 3 model fallback: {len(model_needed)} rows")

    if len(model_needed) > 0:
        X_train = train.drop(columns=[TARGET_COL])
        y_train = train[TARGET_COL].values - 1
        X_train_fe, fit_params = engineer_features(X_train.copy(), is_train=True)
        X_test_fe = apply_features(test.copy(), fit_params)
        full_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        full_model.fit(X_train_fe, y_train, callbacks=[lgb.log_evaluation(0)])
        model_raw = full_model.predict(X_test_fe) + 1
        preds[model_needed] = model_raw[model_needed]
        sources[model_needed] = "tier3_model"

    assert (preds != -1).all(), "Some rows unresolved!"
    assert (preds >= 1).all() and (preds <= 5).all()

    submission = pd.DataFrame({"patient_id": test["patient_id"], "triage_acuity": preds.astype(int)})
    assert list(submission.columns) == list(sample_sub.columns)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out = f"{SUBMISSION_DIR}/submission_final.csv"
    submission.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Distribution:\n{submission['triage_acuity'].value_counts().sort_index()}")
    print(f"\nSources: tier1={(sources == 'tier1').sum()}, tier2={(sources == 'tier2_glaucoma').sum()}, tier3={(sources == 'tier3_model').sum()}")


if __name__ == "__main__":
    main()
