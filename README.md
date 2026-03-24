# Triagegeist — Emergency Triage Acuity Prediction

**Competition:** [Triagegeist on Kaggle](https://www.kaggle.com/competitions/triagegeist) · Community Hackathon · $10,000 prize
**Result:** CV accuracy **0.9995** · 3-tier hybrid prediction system
**Stack:** Python, LightGBM, scikit-learn, Streamlit

---

## The Problem

Given clinical data from an emergency department visit — vitals, demographics, chief complaint, medical history — predict the triage acuity level (1–5, where 1 is most critical).

Three data sources:
- `train.csv` / `test.csv` — 80k/20k rows, 37 clinical features
- `chief_complaints.csv` — raw free-text complaint per patient
- `patient_history.csv` — 24 binary comorbidity flags

---

## What We Found

### The chief complaint text does most of the work

Scaling `chief_complaint_raw` from 50 TF-IDF features to 2000 bigrams moved CV accuracy from 0.891 to 0.9989. One change, 10 points.

The intuition holds up clinically — "thunderclap headache" and "minor skin rash" have very different acuity implications regardless of what the vitals say.

| Experiment | CV Accuracy | Change |
|---|---|---|
| Baseline (TF-IDF 50 features) | 0.8910 | — |
| TF-IDF 150 features | 0.9836 | +0.0926 |
| TF-IDF 300 features | 0.9919 | +0.0083 |
| TF-IDF 500 features | 0.9948 | +0.0029 |
| TF-IDF 1000 features | 0.9980 | +0.0032 |
| TF-IDF 2000 features | 0.9989 | +0.0009 |
| Hyperparameter tuning | 0.9980 | -0.0009 (reverted) |

### Every remaining error traces to one complaint

After the model plateaued at 0.9989, we ran error analysis across all 5 CV folds. Every single one of the 39 errors came from the same source: 15 variants of **"acute angle closure glaucoma"** — a condition that sits right on the clinical boundary between acuity 1 (critical) and acuity 2 (urgent).

The complaint text alone can't resolve it. The vital signs can — specifically the combination of NEWS2 score, GCS total, and pain score.

---

## The Solution: 3-Tier Hybrid Predictor

```
                    test row
                       │
          ┌────────────▼────────────┐
          │  Complaint text in      │
          │  unambiguous lookup?    │
          └──────┬──────────────────┘
                 │ YES (19,885 rows, 99.4%)
                 ▼
          Return label directly ──────────────► prediction
                 │ NO
                 ▼
          ┌──────────────────────────┐
          │  Glaucoma variant?       │
          │  (15 ambiguous texts)    │
          └──────┬───────────────────┘
                 │ YES (76 rows)
                 ▼
          Binary LightGBM
          (news2, gcs, pain, hr) ──────────────► prediction
                 │ NO
                 ▼
          ┌──────────────────────────┐
          │  Unseen complaint text   │
          │  (39 rows)               │
          └──────┬───────────────────┘
                 │
                 ▼
          Full LightGBM multiclass ────────────► prediction
```

**Tier 1 — Direct lookup (19,885 rows)**
4,934 complaint texts in training data that always map to exactly one acuity. For these, no model is needed. The lookup is deterministic and effectively perfect.

**Tier 2 — Glaucoma binary classifier (76 rows)**
Trained only on the 237 "acute angle closure glaucoma" training rows. Features: NEWS2 score, GCS total, pain score, heart rate, systolic BP, respiratory rate, SpO2, temperature, shock index. CV accuracy: **94%** — compared to ~27% (random on imbalanced data) without it.

**Tier 3 — Full LightGBM multiclass (39 rows)**
Complaint texts not seen during training. Falls back to the full feature set including TF-IDF, comorbidities, and all vitals.

**Final CV accuracy: 0.9995**

---

## Feature Engineering

All features live in `src/features.py`. Key design decisions:

**fit_params pattern** — encoders are fit on training data only and applied to validation/test. No leakage.

```python
X_train_fe, fit_params = engineer_features(X_train, is_train=True)
X_val_fe = apply_features(X_val, fit_params)
```

**What we use:**
- TF-IDF bigrams on `chief_complaint_raw` (2000 features, sublinear TF)
- Frequency encoding for all categoricals
- Median imputation for missing vitals (fitted on train)
- Clinical interactions: `gcs × news2`, `resp × spo2`, `pain × news2`
- 24 binary comorbidity flags from `patient_history.csv`
- Comorbidity burden sum

**What we drop:**
- `ed_los_hours` and `disposition` — post-triage outcomes, not in test (leakage)
- `triage_nurse_id`, `site_id` — high cardinality, won't generalise

---

## Repo Structure

```
triagegeist-solution/
├── src/
│   ├── config.py                     # Paths, hyperparams, feature lists
│   ├── features.py                   # Feature engineering
│   ├── pipeline.py                   # LightGBM baseline
│   ├── evaluate.py                   # Locked 5-fold CV evaluator
│   └── generate_submission_final.py  # 3-tier hybrid predictor
├── notebooks/
│   ├── exploration.ipynb             # EDA and signal discovery
│   └── solution.ipynb                # Full Kaggle solution notebook
├── dashboard/
│   └── dashboard.py                  # Streamlit experiment tracker
├── docs/
│   ├── index.html                    # Terminal-styled project page
│   └── thumbnail.html               # Kaggle writeup thumbnail
├── tests/
│   └── test_features.py             # Feature engineering + config tests
├── experiments.jsonl                 # Full experiment audit trail
└── requirements.txt
```

---

## Running It

```bash
pip install -r requirements.txt

# Download competition data from Kaggle into data/
# kaggle competitions download -c triagegeist -p data/

# Run CV evaluation
cd src && python evaluate.py

# Generate submission
cd src && python generate_submission_final.py

# Launch dashboard
streamlit run dashboard/dashboard.py
```

---

## Lessons

**The complaint text does most of the work.** When a dataset has free-text that directly describes what you're predicting, that's the primary feature. Treat it that way from day one, not after everything else has failed.

**Error analysis beats hyperparameter tuning.** Going from 0.9989 to 0.9995 required understanding *why* the model was wrong, not trying parameter combinations. Every single error had the same root cause. Once you find it, the fix is obvious.

**Sometimes the answer is already in the training data.** For 99.4% of this dataset, the right prediction was just a lookup. A lookup can't be wrong the way a model can. The model is only needed for the 0.6% where training data gives no definitive answer.

---

## Competition

[Triagegeist](https://www.kaggle.com/competitions/triagegeist) — Predict emergency severity and optimize triage decisions with AI, powered by clinical data from the Laitinen-Fredriksson Foundation. Deadline: April 21, 2026.

