#!/usr/bin/env python
"""Triagegeist Autoresearch Dashboard — live experiment tracker."""
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Triagegeist Autoresearch", page_icon="🏥", layout="wide")

st.title("🏥 Triagegeist — Autoresearch Dashboard")
st.caption("Live experiment tracker · Kaggle competition · $10,000 prize")

# Load experiments
records = []
import os
EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__), "experiments.jsonl")

try:
    with open(EXPERIMENTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
except FileNotFoundError:
    st.warning("No experiments.jsonl found yet. Run evaluate.py to start.")
    st.stop()

df = pd.DataFrame(records)
cv_runs = df[df["type"] == "cv_run"].copy() if "type" in df.columns else df.copy()

if cv_runs.empty:
    st.info("No CV runs yet.")
    st.stop()

cv_runs["timestamp"] = pd.to_datetime(cv_runs["timestamp"])
cv_runs = cv_runs.sort_values("timestamp").reset_index(drop=True)
cv_runs["run_num"] = range(1, len(cv_runs) + 1)

best_score = cv_runs["cv_score"].max()
latest_score = cv_runs["cv_score"].iloc[-1]
n_runs = len(cv_runs)

# ── Metrics row ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best CV Score", f"{best_score:.4f}")
col2.metric("Latest CV Score", f"{latest_score:.4f}",
            delta=f"{latest_score - cv_runs['cv_score'].iloc[-2]:.4f}" if n_runs > 1 else None)
col3.metric("Total Runs", n_runs)
col4.metric("Target (LB)", "1.0000")

st.divider()

# ── Score over time ────────────────────────────────────────────────────────
st.subheader("📈 CV Score Over Runs")
chart_df = cv_runs[["run_num", "cv_score"]].set_index("run_num")
st.line_chart(chart_df, y="cv_score", height=300)

# ── Experiment table ───────────────────────────────────────────────────────
st.subheader("📋 All Experiments")
display_cols = ["run_num", "timestamp", "cv_score", "cv_std"]
if "status" in cv_runs.columns:
    display_cols.append("status")
st.dataframe(
    cv_runs[display_cols].style.highlight_max(subset=["cv_score"], color="#1a472a"),
    use_container_width=True,
    hide_index=True,
)

# ── Latest fold breakdown ──────────────────────────────────────────────────
if "fold_scores" in cv_runs.columns:
    latest = cv_runs.iloc[-1]
    if isinstance(latest["fold_scores"], list):
        st.subheader("🔍 Latest Run — Fold Breakdown")
        fold_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(latest["fold_scores"]))],
            "Score": latest["fold_scores"]
        })
        st.bar_chart(fold_df.set_index("Fold"), height=250)

st.caption("Auto-refreshes every 30 seconds · Data from experiments.jsonl")
st.button("🔄 Refresh now")
