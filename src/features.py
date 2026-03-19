"""
Feature engineering for Triagegeist.
Merges: main table + chief_complaints + patient_history.
fit_params pattern: engineer_features(df, is_train=True) fits encoders,
                    apply_features(df, fit_params) applies them to val/test.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (COMPLAINTS_PATH, HISTORY_PATH, LEAKAGE_COLS, DROP_COLS,
                    VITAL_COLS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF)


def _load_aux(df):
    """Merge chief_complaints and patient_history onto main dataframe."""
    complaints = pd.read_csv(COMPLAINTS_PATH)
    history = pd.read_csv(HISTORY_PATH)

    df = df.merge(complaints[["patient_id", "chief_complaint_raw"]], on="patient_id", how="left")
    df = df.merge(history, on="patient_id", how="left")
    return df


def engineer_features(df, is_train=True, fit_params=None):
    if fit_params is None:
        fit_params = {}

    # ── 1. Merge auxiliary tables ───────────────────────────────────────────
    df = _load_aux(df)

    # ── 2. Drop leakage + unwanted ID columns ──────────────────────────────
    drop = LEAKAGE_COLS + DROP_COLS + ["patient_id"]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # ── 3. Impute missing vitals ────────────────────────────────────────────
    for col in VITAL_COLS:
        if col in df.columns:
            if is_train:
                fit_params[f"median_{col}"] = df[col].median()
            df[col] = df[col].fillna(fit_params.get(f"median_{col}", 0))

    # ── 4. Interaction / derived features ──────────────────────────────────
    if "respiratory_rate" in df.columns and "spo2" in df.columns:
        df["resp_x_spo2"] = df["respiratory_rate"] * df["spo2"]
    if "heart_rate" in df.columns and "systolic_bp" in df.columns:
        df["hr_x_sbp"] = df["heart_rate"] * df["systolic_bp"]
    if "gcs_total" in df.columns and "news2_score" in df.columns:
        df["gcs_x_news2"] = df["gcs_total"] * df["news2_score"]
    if "pain_score" in df.columns and "news2_score" in df.columns:
        df["pain_x_news2"] = df["pain_score"] * df["news2_score"]
    if "num_prior_ed_visits_12m" in df.columns and "num_comorbidities" in df.columns:
        df["ed_visits_x_comorbid"] = df["num_prior_ed_visits_12m"] * df["num_comorbidities"]

    # Comorbidity burden
    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    if hx_cols:
        df["total_comorbidities"] = df[hx_cols].sum(axis=1)

    # ── 5. NLP: TF-IDF on chief complaint text ─────────────────────────────
    text_col = "chief_complaint_raw"
    if text_col in df.columns:
        df[text_col] = df[text_col].fillna("unknown")
        if is_train:
            tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                    ngram_range=TFIDF_NGRAM_RANGE,
                                    sublinear_tf=True, min_df=TFIDF_MIN_DF)
            tfidf.fit(df[text_col])
            fit_params["tfidf"] = tfidf
        tfidf = fit_params.get("tfidf")
        if tfidf is not None:
            tfidf_mat = tfidf.transform(df[text_col]).toarray()
            tfidf_df = pd.DataFrame(tfidf_mat,
                                    columns=[f"tfidf_{i}" for i in range(tfidf_mat.shape[1])],
                                    index=df.index)
            df = pd.concat([df, tfidf_df], axis=1)
        df = df.drop(columns=[text_col])

    # ── 6. Encode categoricals ──────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if is_train:
            freq = df[col].value_counts(normalize=True)
            fit_params[f"freq_{col}"] = freq.to_dict()
        freq_map = fit_params.get(f"freq_{col}", {})
        df[col + "_enc"] = df[col].map(freq_map).fillna(0).astype(float)
    df = df.drop(columns=cat_cols)

    return df, fit_params


def apply_features(df, fit_params):
    df, _ = engineer_features(df, is_train=False, fit_params=fit_params)
    return df
