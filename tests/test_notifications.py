"""
test_notifications.py — Tests for notifications.py
Covers: dashboard notification store, mark read, notification types.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestDashboardNotifications:
    """Test dashboard notification store."""

    def test_add_dashboard_notification(self, tmp_path):
        import notifications
        original_log = notifications.NOTIF_LOG
        notifications.NOTIF_LOG = str(tmp_path / "notifications.json")
        notifications._dashboard_notifications = []

        entry = notifications._add_dashboard_notification(
            "new_lead", "Test Title", "Test Message",
            customer_name="Priya", lead_id="L00001"
        )
        notifications.NOTIF_LOG = original_log
        assert entry["type"] == "new_lead"
        assert entry["title"] == "Test Title"
        assert entry["message"] == "Test Message"
        assert entry["customer_name"] == "Priya"
        assert entry["lead_id"] == "L00001"
        assert entry["read"] is False

    def test_get_unread_notifications(self, tmp_path):
        import notifications
        original_log = notifications.NOTIF_LOG
        notifications.NOTIF_LOG = str(tmp_path / "notifications.json")
        notifications._dashboard_notifications = []

        notifications._add_dashboard_notification("info", "Read Me", "msg1")
        notifications._add_dashboard_notification("info", "Unread", "msg2")

        unread = notifications.get_unread_notifications()
        notifications.NOTIF_LOG = original_log
        assert len(unread) == 2

    def test_mark_notification_read(self, tmp_path):
        import notifications
        original_log = notifications.NOTIF_LOG
        notifications.NOTIF_LOG = str(tmp_path / "notifications.json")
        notifications._dashboard_notifications = []

        notifications._add_dashboard_notification("info", "Test", "msg")
        notifications.mark_notification_read(0)

        unread = notifications.get_unread_notifications()
        notifications.NOTIF_LOG = original_log
        assert len(unread) == 0

    def test_mark_all_read(self, tmp_path):
        import notifications
        original_log = notifications.NOTIF_LOG
        notifications.NOTIF_LOG = str(tmp_path / "notifications.json")
        notifications._dashboard_notifications = []

        notifications._add_dashboard_notification("info", "Test1", "msg1")
        notifications._add_dashboard_notification("info", "Test2", "msg2")
        notifications.mark_all_read()

        unread = notifications.get_unread_notifications()
        notifications.NOTIF_LOG = original_log
        assert len(unread) == 0

    def test_notification_types(self, tmp_path):
        import notifications
        original_log = notifications.NOTIF_LOG
        notifications.NOTIF_LOG = str(tmp_path / "notifications.json")
        notifications._dashboard_notifications = []

        notifications._add_dashboard_notification("new_lead", "New", "msg")
        notifications._add_dashboard_notification("auto_reply", "Reply", "msg")
        notifications._add_dashboard_notification("overdue", "Overdue", "msg")
        notifications._add_dashboard_notification("info", "Info", "msg")

        notifs = notifications.get_unread_notifications()
        notifications.NOTIF_LOG = original_log
        types = [n["type"] for n in notifs]
        assert "new_lead" in types
        assert "auto_reply" in types
        assert "overdue" in types
        assert "info" in types


class TestNotificationPublicFunctions:
    """Test public notification functions (notify_new_lead, etc.)."""

    def test_notify_new_lead_creates_entry(self, tmp_path):
        import notifications
        original_log = notifications.NOTIF_LOG
        notifications.NOTIF_LOG = str(tmp_path / "notifications.json")
        notifications._dashboard_notifications = []

        notifications.notify_new_lead("Priya", "priya@test.com", 48.5, 0.87)
        notifs = notifications.get_unread_notifications()
        notifications.NOTIF_LOG = original_log
        assert len(notifs) >= 1
        assert any("Priya" in n.get("customer_name", "") for n in notifs)

    def test_notify_auto_reply_creates_entry(self, tmp_path):
        import notifications
        original_log = notifications.NOTIF_LOG
        notifications.NOTIF_LOG = str(tmp_path / "notifications.json")
        notifications._dashboard_notifications = []

        notifications.notify_auto_reply("Rahul", "rahul@test.com", "pricing")
        notifs = notifications.get_unread_notifications()
        notifications.NOTIF_LOG = original_log
        assert len(notifs) >= 1
