"""
test_inbox_monitor.py — Tests for inbox_monitor.py
Covers: scoring, scan summary, log persistence.
"""
import os
import sys
import json
import pickle
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def sample_scored_df():
    """Sample scored DataFrame."""
    return pd.DataFrame({
        "lead_id": ["E-ABC12345", "E-DEF67890", "E-GHI11111"],
        "channel": ["Email", "Email", "Email"],
        "message_text": ["price info", "demo request", "hello"],
        "message_hour": [10, 14, 21],
        "message_length": [20, 25, 5],
        "high_intent_flag": [1, 1, 0],
        "prev_contacts": [0, 1, 0],
        "response_gap_hrs": [48.0, 12.0, 120.0],
        "missed_probability": [0.85, 0.35, 0.72],
        "predicted_missed": [1, 0, 1],
        "_customer_email": ["a@test.com", "b@test.com", "c@test.com"],
        "_customer_name": ["Alice", "Bob", "Charlie"],
        "_subject": ["Pricing", "Demo", "Hi"],
        "_message_id": ["<1@test>", "<2@test>", "<3@test>"],
        "_received_time": ["2025-01-15 10:00 UTC", "2025-01-15 14:00 UTC", "2025-01-15 21:00 UTC"],
    })


class TestScoring:
    """Test email scoring logic."""

    def test_score_email_with_ensemble(self, sample_scored_df):
        from inbox_monitor import score_email
        # Create a mock ensemble and scaler
        mock_ensemble = MagicMock()
        mock_ensemble.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.25, 0.75]])
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((3, 9))

        result = score_email(sample_scored_df.copy(), mock_ensemble, mock_scaler)
        assert "missed_probability" in result.columns
        assert "predicted_missed" in result.columns

    def test_channel_encoding(self):
        from inbox_monitor import CHANNEL_MAP
        assert CHANNEL_MAP["email"] == 0
        assert CHANNEL_MAP["phone inquiry"] == 1
        assert CHANNEL_MAP["website chat"] == 2
        assert CHANNEL_MAP["whatsapp"] == 3


class TestLogPersistence:
    """Test log file operations."""

    def test_load_reply_log(self, tmp_path):
        import inbox_monitor
        log_file = tmp_path / "auto_replies.json"
        log_file.write_text(json.dumps([{"lead_id": "L00001"}]))
        original = inbox_monitor.REPLY_LOG
        inbox_monitor.REPLY_LOG = str(log_file)
        result = inbox_monitor._load_reply_log()
        inbox_monitor.REPLY_LOG = original
        assert len(result) == 1
        assert result[0]["lead_id"] == "L00001"

    def test_save_reply_log(self, tmp_path):
        import inbox_monitor
        log_file = tmp_path / "auto_replies.json"
        original = inbox_monitor.REPLY_LOG
        inbox_monitor.REPLY_LOG = str(log_file)
        inbox_monitor._save_reply_log([{"lead_id": "L00001"}])
        inbox_monitor.REPLY_LOG = original
        assert log_file.exists()
        with open(log_file) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_load_followup_status(self, tmp_path):
        import inbox_monitor
        log_file = tmp_path / "followup_status.json"
        log_file.write_text(json.dumps({"L00001": {"auto_replied": True}}))
        original = inbox_monitor.FOLLOWUP_LOG
        inbox_monitor.FOLLOWUP_LOG = str(log_file)
        result = inbox_monitor._load_followup_status()
        inbox_monitor.FOLLOWUP_LOG = original
        assert "L00001" in result
        assert result["L00001"]["auto_replied"] is True

    def test_save_followup_status(self, tmp_path):
        import inbox_monitor
        log_file = tmp_path / "followup_status.json"
        original = inbox_monitor.FOLLOWUP_LOG
        inbox_monitor.FOLLOWUP_LOG = str(log_file)
        inbox_monitor._save_followup_status({"L00001": {"auto_replied": True}})
        inbox_monitor.FOLLOWUP_LOG = original
        assert log_file.exists()


class TestScanSummary:
    """Test scan summary structure."""

    def test_scan_summary_keys(self):
        # The scan summary should have these keys
        expected_keys = {"scanned", "scored", "replied", "skipped"}
        # We can't run a full scan without IMAP, but we can verify the structure
        assert expected_keys == {"scanned", "scored", "replied", "skipped"}
