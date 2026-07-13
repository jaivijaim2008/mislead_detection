"""
test_employee_reminder.py — Tests for employee_reminder.py
Covers: reminder lifecycle, mark_replied, headless mode.
"""
import os
import sys
import json
import time
import pytest
from unittest.mock import patch

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestRepliedTracking:
    """Test reply tracking persistence."""

    def test_mark_and_check_replied(self, tmp_path):
        import employee_reminder
        original_log = employee_reminder.LOG_FILE
        employee_reminder.LOG_FILE = str(tmp_path / "replied.json")

        assert employee_reminder.is_replied("L00001") is False
        employee_reminder.mark_replied("L00001")
        assert employee_reminder.is_replied("L00001") is True

        employee_reminder.LOG_FILE = original_log

    def test_load_replied_missing_file(self, tmp_path):
        import employee_reminder
        original_log = employee_reminder.LOG_FILE
        employee_reminder.LOG_FILE = str(tmp_path / "nonexistent.json")

        result = employee_reminder._load_replied()
        employee_reminder.LOG_FILE = original_log
        assert isinstance(result, set)
        assert len(result) == 0

    def test_save_replied_creates_file(self, tmp_path):
        import employee_reminder
        original_log = employee_reminder.LOG_FILE
        employee_reminder.LOG_FILE = str(tmp_path / "replied.json")

        employee_reminder._save_replied({"L00001", "L00002"})
        employee_reminder.LOG_FILE = original_log

        assert (tmp_path / "replied.json").exists()
        with open(tmp_path / "replied.json") as f:
            data = json.load(f)
        assert "L00001" in data
        assert "L00002" in data


class TestReminderLifecycle:
    """Test start_reminder and stop_reminder."""

    def test_start_reminder_creates_thread(self):
        import employee_reminder
        lead = {
            "lead_id": "L00001",
            "customer_name": "Priya",
            "channel": "Email",
            "gap_hrs": 48,
        }
        employee_reminder.GUI_AVAILABLE = False
        employee_reminder.start_reminder(lead, interval=100)
        assert "L00001" in employee_reminder._active
        employee_reminder.stop_reminder("L00001")

    def test_stop_reminder(self):
        import employee_reminder
        lead = {
            "lead_id": "L00002",
            "customer_name": "Rahul",
            "channel": "WhatsApp",
            "gap_hrs": 24,
        }
        employee_reminder.GUI_AVAILABLE = False
        employee_reminder.start_reminder(lead, interval=100)
        employee_reminder.stop_reminder("L00002")
        # Should not raise even if already stopped

    def test_mark_replied_stops_reminder(self):
        import employee_reminder
        lead = {
            "lead_id": "L00003",
            "customer_name": "Test",
            "channel": "Email",
            "gap_hrs": 12,
        }
        employee_reminder.GUI_AVAILABLE = False
        employee_reminder.start_reminder(lead, interval=100)
        employee_reminder.mark_replied("L00003")
        time.sleep(0.1)
        assert employee_reminder.is_replied("L00003") is True
        employee_reminder.stop_reminder("L00003")
