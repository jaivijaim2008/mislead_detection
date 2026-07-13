"""
test_email_reader.py — Tests for email_reader.py
Covers: email filtering, feature extraction, helper functions.
"""
import os
import sys
import re
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestEmailFiltering:
    """Test email filtering functions."""

    def test_is_promotional_sender_domain(self):
        from email_reader import _is_promotional_email
        assert _is_promotional_email("newsletter@mailchimp.com", "Newsletter", "") is True
        assert _is_promotional_email("info@facebookmail.com", "Update", "") is True
        assert _is_promotional_email("test@gmail.com", "Hello", "") is False

    def test_is_promotional_subject_keyword(self):
        from email_reader import _is_promotional_email
        assert _is_promotional_email("test@test.com", "Weekly Digest Update", "") is True
        assert _is_promotional_email("test@test.com", "Don't Miss This Sale", "") is True
        assert _is_promotional_email("test@test.com", "Course Pricing", "") is False

    def test_is_promotional_body_keywords(self):
        from email_reader import _is_promotional_email
        assert _is_promotional_email(
            "test@test.com", "Update", "Click here to unsubscribe from our list"
        ) is True
        assert _is_promotional_email(
            "test@test.com", "Hello", "Thanks for reaching out!"
        ) is False

    def test_is_automated_otp_subject(self):
        from email_reader import _is_automated_notification
        assert _is_automated_notification("test@test.com", "Your OTP is 123456", "") is True
        assert _is_automated_notification("test@test.com", "Verification code: ABC123", "") is True
        assert _is_automated_notification("test@test.com", "Course pricing info", "") is False

    def test_is_automated_banking(self):
        from email_reader import _is_automated_notification
        assert _is_automated_notification("test@test.com", "Transaction Alert: Rs 5000", "") is True
        assert _is_automated_notification("test@test.com", "Payment Confirmation", "") is True

    def test_is_automated_social(self):
        from email_reader import _is_automated_notification
        assert _is_automated_notification("test@test.com", "You have a new follower", "") is True
        assert _is_automated_notification("test@test.com", "Someone liked your post", "") is True

    def test_is_automated_sender_patterns(self):
        from email_reader import _is_automated_notification
        assert _is_automated_notification("otp@system.com", "Alert", "") is True
        assert _is_automated_notification("alert@bank.com", "Notice", "") is True
        assert _is_automated_notification("sales@company.com", "Pricing", "") is False

    def test_is_auto_reply_headers(self):
        from email_reader import _is_auto_reply
        class MockMsg:
            def __init__(self, headers):
                self._headers = headers
            def get(self, key, default=""):
                return self._headers.get(key, default)

        msg = MockMsg({"Auto-Submitted": "auto-replied"})
        assert _is_auto_reply(msg) is True

        msg2 = MockMsg({"Subject": "Out of Office"})
        assert _is_auto_reply(msg2) is True

        msg3 = MockMsg({"Subject": "Course Pricing"})
        assert _is_auto_reply(msg3) is False


class TestEmailHelpers:
    """Test helper functions."""

    def test_decode_str_handles_none(self):
        from email_reader import _decode_str
        assert _decode_str(None) == ""

    def test_decode_str_handles_string(self):
        from email_reader import _decode_str
        assert _decode_str("Hello World") == "Hello World"

    def test_extract_email_address_angle_brackets(self):
        from email_reader import _extract_email_address
        result = _extract_email_address("Priya Sharma <priya@gmail.com>")
        assert result == "priya@gmail.com"

    def test_extract_email_address_no_brackets(self):
        from email_reader import _extract_email_address
        result = _extract_email_address("priya@gmail.com")
        assert result == "priya@gmail.com"

    def test_compute_lead_id_deterministic(self):
        from email_reader import _compute_lead_id
        id1 = _compute_lead_id("test@gmail.com", "Subject", "2025-01-01")
        id2 = _compute_lead_id("test@gmail.com", "Subject", "2025-01-01")
        assert id1 == id2
        assert id1.startswith("E-")
        assert len(id1) == 10  # E- + 8 hex chars

    def test_compute_lead_id_different_inputs(self):
        from email_reader import _compute_lead_id
        id1 = _compute_lead_id("a@gmail.com", "Subject A", "2025-01-01")
        id2 = _compute_lead_id("b@gmail.com", "Subject B", "2025-01-02")
        assert id1 != id2

    def test_is_our_own_email(self):
        from email_reader import _is_our_own_email
        # Should return True for empty sender
        assert _is_our_own_email("") is True
        # Should return False for test inquiry emails
        assert _is_our_own_email("test@gmail.com", "[TEST] Inquiry") is False
