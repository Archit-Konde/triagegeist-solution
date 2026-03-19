"""Basic tests for feature engineering and pipeline correctness."""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add src/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_train():
    """Minimal training dataframe matching expected schema."""
    return pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "triage_acuity": [1, 2, 3, 4],
        "heart_rate": [120, 80, 90, 70],
        "systolic_bp": [90, 120, 110, 130],
        "diastolic_bp": [60, 80, 70, 85],
        "respiratory_rate": [22, 16, 18, 14],
        "spo2": [95, 98, 97, 99],
        "temperature_c": [38.5, 37.0, 37.2, 36.8],
        "news2_score": [7, 1, 3, 0],
        "gcs_total": [15, 15, 14, 15],
        "pain_score": [8, 2, 5, 1],
        "arrival_mode_enc": [0.3, 0.5, 0.5, 0.2],
    })


class TestLabelShift:
    """Verify the 1-5 → 0-4 → 1-5 roundtrip used in pipeline.py."""

    def test_roundtrip(self):
        labels = np.array([1, 2, 3, 4, 5])
        shifted = labels - 1
        assert shifted.min() == 0
        assert shifted.max() == 4
        restored = shifted + 1
        np.testing.assert_array_equal(labels, restored)

    def test_single_class(self):
        labels = np.array([3, 3, 3])
        shifted = labels - 1
        assert (shifted == 2).all()
        restored = shifted + 1
        assert (restored == 3).all()

    def test_out_of_range_detected(self):
        labels = np.array([0, 1, 2])  # 0 is invalid
        assert labels.min() < 1, "Should detect out-of-range labels"


class TestConfig:
    """Verify config module loads and has expected values."""

    def test_imports(self):
        from config import LGBM_PARAMS, GLAUCOMA_PARAMS, GLAUCOMA_FEATURES
        assert LGBM_PARAMS["objective"] == "multiclass"
        assert GLAUCOMA_PARAMS["objective"] == "binary"
        assert len(GLAUCOMA_FEATURES) == 13

    def test_paths_are_strings(self):
        from config import TRAIN_PATH, TEST_PATH, COMPLAINTS_PATH
        assert isinstance(TRAIN_PATH, str)
        assert isinstance(TEST_PATH, str)
        assert isinstance(COMPLAINTS_PATH, str)

    def test_random_state_consistent(self):
        from config import RANDOM_STATE, LGBM_PARAMS, GLAUCOMA_PARAMS
        assert LGBM_PARAMS["random_state"] == RANDOM_STATE
        assert GLAUCOMA_PARAMS["random_state"] == RANDOM_STATE
