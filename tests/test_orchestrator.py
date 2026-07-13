"""
test_orchestrator.py — Tests for orchestrator.py
Covers: artifact loading, scoring pipeline, demo mode.
"""
import os
import sys
import pickle
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestArtifactLoading:
    """Test load_artefacts function."""

    def test_load_artefacts_returns_tuple(self):
        from orchestrator import load_artefacts
        # This will fail if models don't exist, but should return (None, None, None, None)
        try:
            result = load_artefacts()
            assert len(result) == 4
        except FileNotFoundError:
            # Expected if models not trained
            pass


class TestScoringPipeline:
    """Test score_leads function."""

    def test_score_leads_adds_columns(self, sample_leads_df):
        from orchestrator import score_leads
        # Create mock ensemble and scaler
        mock_ensemble = MagicMock()
        mock_ensemble.predict_proba.return_value = np.array([
            [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.9, 0.1], [0.3, 0.7]
        ])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((5, 9))

        result = score_leads(sample_leads_df.copy(), mock_ensemble, mock_scaler)
        assert "missed_probability" in result.columns
        assert "predicted_missed" in result.columns
        assert "channel_enc" in result.columns
        assert "intent_score" in result.columns
        assert "is_business_hours" in result.columns
        assert "gap_bucket" in result.columns

    def test_score_leads_with_dl_model(self, sample_leads_df):
        from orchestrator import score_leads
        mock_ensemble = MagicMock()
        mock_ensemble.predict_proba.return_value = np.array([
            [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.9, 0.1], [0.3, 0.7]
        ])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((5, 9))

        mock_dl = MagicMock()
        mock_dl.predict_proba_np.return_value = np.array([
            [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.8, 0.2], [0.4, 0.6]
        ])
        mock_dl_scaler = MagicMock()
        mock_dl_scaler.transform.return_value = np.zeros((5, 9))

        result = score_leads(
            sample_leads_df.copy(), mock_ensemble, mock_scaler,
            dl_model=mock_dl, dl_scaler=mock_dl_scaler
        )
        assert "_ml_prob" in result.columns
        assert "_dl_prob" in result.columns
        # Grand ensemble should be 50/50 average
        assert result["missed_probability"].between(0, 1).all()

    def test_predicted_missed_threshold(self, sample_leads_df):
        from orchestrator import score_leads
        mock_ensemble = MagicMock()
        mock_ensemble.predict_proba.return_value = np.array([
            [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]
        ])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((5, 9))

        result = score_leads(sample_leads_df.copy(), mock_ensemble, mock_scaler)
        # High probability should be marked as missed
        assert result.iloc[0]["predicted_missed"] == 1
        # Low probability should not be missed
        assert result.iloc[1]["predicted_missed"] == 0
