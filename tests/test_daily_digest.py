"""
test_daily_digest.py — Tests for daily_digest.py
Covers: digest building, HTML generation, data filtering.
"""
import os
import sys
import json
import pytest
from datetime import datetime, timezone, timedelta

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestDigestBuilding:
    """Test build_digest function."""

    def test_build_digest_returns_dict(self):
        from daily_digest import build_digest
        result = build_digest()
        assert isinstance(result, dict)

    def test_build_digest_has_required_keys(self):
        from daily_digest import build_digest
        result = build_digest()
        required_keys = [
            "generated_at", "date", "emails_scored_24h", "new_leads",
            "auto_replies_sent", "overdue_alerts", "total_tracked_leads",
            "total_auto_replied", "total_human_followed", "still_pending",
            "follow_up_rate", "intent_breakdown", "avg_missed_probability",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_build_digest_date_format(self):
        from daily_digest import build_digest
        result = build_digest()
        # Date should be like "January 15, 2025"
        assert isinstance(result["date"], str)
        assert len(result["date"]) > 5

    def test_build_digest_follow_up_rate(self):
        from daily_digest import build_digest
        result = build_digest()
        assert isinstance(result["follow_up_rate"], (int, float))
        assert 0 <= result["follow_up_rate"] <= 100


class TestFilterLast24h:
    """Test _filter_last_24h function."""

    def test_filters_recent_entries(self):
        from daily_digest import _filter_last_24h
        now = datetime.now(timezone.utc)
        entries = [
            {"timestamp": now.strftime("%Y-%m-%d %H:%M:%S")},
            {"timestamp": (now - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")},
        ]
        result = _filter_last_24h(entries)
        assert len(result) == 1

    def test_handles_empty_entries(self):
        from daily_digest import _filter_last_24h
        result = _filter_last_24h([])
        assert len(result) == 0

    def test_handles_missing_timestamp(self):
        from daily_digest import _filter_last_24h
        entries = [{"type": "info"}]
        result = _filter_last_24h(entries)
        assert len(result) == 0


class TestHTMLGeneration:
    """Test HTML email template generation."""

    def test_build_html_returns_string(self):
        from daily_digest import _build_html
        data = {
            "generated_at": "2025-01-15 10:00 UTC",
            "date": "January 15, 2025",
            "emails_scored_24h": 25,
            "new_leads": 5,
            "auto_replies_sent": 3,
            "overdue_alerts": 1,
            "total_tracked_leads": 10,
            "total_auto_replied": 8,
            "total_human_followed": 4,
            "still_pending": 2,
            "follow_up_rate": 50.0,
            "intent_breakdown": {"pricing": 5, "demo": 3},
            "avg_missed_probability": 0.65,
            "new_leads_details": [{"name": "Priya", "title": "Test"}],
            "overdue_details": [{"name": "Rahul", "title": "Overdue"}],
        }
        html = _build_html(data)
        assert isinstance(html, str)
        assert "<html>" in html
        assert "Daily Inbox Digest" in html
        assert "25" in str(data["emails_scored_24h"])

    def test_build_plain_returns_string(self):
        from daily_digest import _build_plain
        data = {
            "generated_at": "2025-01-15 10:00 UTC",
            "date": "January 15, 2025",
            "emails_scored_24h": 25,
            "new_leads": 5,
            "auto_replies_sent": 3,
            "overdue_alerts": 1,
            "total_tracked_leads": 10,
            "total_auto_replied": 8,
            "total_human_followed": 4,
            "still_pending": 2,
            "follow_up_rate": 50.0,
            "intent_breakdown": {"pricing": 5},
            "avg_missed_probability": 0.65,
            "new_leads_details": [],
            "overdue_details": [],
        }
        plain = _build_plain(data)
        assert isinstance(plain, str)
        assert "DAILY INBOX DIGEST" in plain
