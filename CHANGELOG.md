# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2026-03-17

### Added
- 3-tier hybrid prediction system (lookup + glaucoma binary classifier + LightGBM multiclass)
- TF-IDF bigram feature engineering on `chief_complaint_raw` (2000 features)
- Locked 5-fold stratified CV evaluator (`evaluate.py`)
- Feature engineering with fit_params pattern to prevent leakage (`features.py`)
- Glaucoma-specific binary LightGBM classifier (94% CV accuracy on ambiguous cases)
- Streamlit experiment tracker dashboard (`dashboard/dashboard.py`)
- EDA notebook with target distribution, TF-IDF scaling analysis, error breakdown
- Full experiment audit trail (`experiments.jsonl`)
- GitHub Pages terminal landing page (`docs/index.html`)
- Streamlit Community Cloud deployment
