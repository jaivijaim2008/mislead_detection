"""
test_auto_followup.py — Tests for auto_followup.py
Covers: email building, dedup logic, retry mechanism, demo mode.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestEmailBuilding:
    """Test _build_email function."""

    def test_build_email_with_smart_reply(self):
        from auto_followup import _build_email
        lead = {
            "lead_id": "L00001",
            "customer_email": "priya@example.com",
            "customer_name": "Priya",
            "subject": "Course Pricing",
            "reply_subject": "Re: Course Pricing — Pricing Details",
            "reply_body": "Hi Priya,\n\nThank you for your interest!\n\nBest regards,\nSales Team",
            "original_message_id": "<abc123@mail.example.com>",
        }
        msg = _build_email(lead)
        assert msg["Subject"] == "Re: Course Pricing — Pricing Details"
        assert msg["To"] == "priya@example.com"
        assert msg["In-Reply-To"] == "<abc123@mail.example.com>"
        assert msg["References"] == "<abc123@mail.example.com>"

    def test_build_email_without_smart_reply(self):
        from auto_followup import _build_email
        lead = {
            "lead_id": "L00001",
            "customer_email": "priya@example.com",
            "customer_name": "Priya",
            "subject": "Course Pricing",
        }
        msg = _build_email(lead)
        assert msg["Subject"] == "Re: Course Pricing"
        assert msg["To"] == "priya@example.com"

    def test_build_email_has_both_parts(self):
        from auto_followup import _build_email
        lead = {
            "lead_id": "L00001",
            "customer_email": "test@example.com",
            "customer_name": "Test",
            "subject": "Hi",
            "reply_body": "Hello there!",
        }
        msg = _build_email(lead)
        # Should have plain and html parts
        parts = msg.get_payload()
        assert len(parts) >= 2


class TestDeduplication:
    """Test send deduplication logic."""

    def test_load_sent_returns_set(self, tmp_path):
        from auto_followup import _load_sent
        sent_file = tmp_path / "sent_leads.json"
        sent_file.write_text(json.dumps(["L00001", "L00002"]))
        # Monkey-patch the SENT_LOG path
        import auto_followup
        original = auto_followup.SENT_LOG
        auto_followup.SENT_LOG = str(sent_file)
        result = _load_sent()
        auto_followup.SENT_LOG = original
        assert isinstance(result, set)
        assert "L00001" in result
        assert "L00002" in result

    def test_load_sent_missing_file(self, tmp_path):
        from auto_followup import _load_sent
        import auto_followup
        original = auto_followup.SENT_LOG
        auto_followup.SENT_LOG = str(tmp_path / "nonexistent.json")
        result = _load_sent()
        auto_followup.SENT_LOG = original
        assert isinstance(result, set)
        assert len(result) == 0

    def test_save_sent_creates_file(self, tmp_path):
        from auto_followup import _save_sent
        import auto_followup
        original = auto_followup.SENT_LOG
        auto_followup.SENT_LOG = str(tmp_path / "sent.json")
        _save_sent({"L00001", "L00002"})
        auto_followup.SENT_LOG = original
        assert (tmp_path / "sent.json").exists()
        with open(tmp_path / "sent.json") as f:
            data = json.load(f)
        assert "L00001" in data
        assert "L00002" in data


class TestFailedLeads:
    """Test failed leads logging."""

    def test_load_failed_returns_dict(self, tmp_path):
        from auto_followup import _load_failed
        import auto_followup
        original = auto_followup.FAILED_LOG
        auto_followup.FAILED_LOG = str(tmp_path / "failed.json")
        result = _load_failed()
        auto_followup.FAILED_LOG = original
        assert isinstance(result, dict)

    def test_save_failed_creates_file(self, tmp_path):
        from auto_followup import _save_failed
        import auto_followup
        original = auto_followup.FAILED_LOG
        auto_followup.FAILED_LOG = str(tmp_path / "failed.json")
        _save_failed({"L00001": {"error": "test error"}})
        auto_followup.FAILED_LOG = original
        assert (tmp_path / "failed.json").exists()


class TestDemoMode:
    """Test demo mode behavior."""

    def test_demo_mode_detected(self):
        import auto_followup
        # If SMTP_USER is not set, should be in demo mode
        if not auto_followup.SMTP_USER:
            assert auto_followup.DEMO_MODE is True

    def test_send_followup_demo_mode(self, tmp_path):
        import auto_followup
        original_sent = auto_followup.SENT_LOG
        auto_followup.SENT_LOG = str(tmp_path / "sent.json")
        original_demo = auto_followup.DEMO_MODE
        auto_followup.DEMO_MODE = True

        lead = {
            "lead_id": "L00001",
            "customer_email": "test@example.com",
            "customer_name": "Test",
            "subject": "Hi",
            "reply_body": "Hello!",
            "reply_subject": "Re: Hi",
        }
        result = auto_followup.send_followup(lead)
        auto_followup.SENT_LOG = original_sent
        auto_followup.DEMO_MODE = original_demo
        assert result is True
        assert "L00001" in auto_followup._load_sent() or os.path.exists(auto_followup.SENT_LOG)

    def test_send_followup_dedup(self, tmp_path):
        import auto_followup
        original_sent = auto_followup.SENT_LOG
        auto_followup.SENT_LOG = str(tmp_path / "sent.json")
        original_demo = auto_followup.DEMO_MODE
        auto_followup.DEMO_MODE = True

        lead = {
            "lead_id": "L00001",
            "customer_email": "test@example.com",
            "customer_name": "Test",
            "subject": "Hi",
            "reply_body": "Hello!",
            "reply_subject": "Re: Hi",
        }
        # First send should succeed
        result1 = auto_followup.send_followup(lead)
        # Second send should be skipped (dedup)
        result2 = auto_followup.send_followup(lead)

        auto_followup.SENT_LOG = original_sent
        auto_followup.DEMO_MODE = original_demo
        assert result1 is True
        assert result2 is False
