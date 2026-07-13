"""
test_database.py — Tests for database.py
Covers: all CRUD operations, schema creation, stats, dedup.
"""
import os
import sys
import pytest
import tempfile

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from database import Database, reset_db


@pytest.fixture
def db(tmp_path):
    """Create a fresh test database."""
    reset_db()
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    yield database
    reset_db()


class TestSchemaCreation:
    """Test database schema initialization."""

    def test_creates_database_file(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        database = Database(db_path)
        assert os.path.exists(db_path)

    def test_creates_all_tables(self, db):
        with db._conn() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t["name"] for t in tables}
            assert "leads" in table_names
            assert "replies" in table_names
            assert "followup_status" in table_names
            assert "notifications" in table_names
            assert "sent_leads" in table_names
            assert "failed_leads" in table_names
            assert "scan_log" in table_names
            assert "seen_email_ids" in table_names

    def test_creates_indexes(self, db):
        with db._conn() as conn:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            index_names = {i["name"] for i in indexes}
            assert "idx_leads_predicted" in index_names


class TestLeads:
    """Test lead CRUD operations."""

    def test_insert_lead(self, db):
        lead = {
            "lead_id": "L00001",
            "channel": "Email",
            "message_text": "What is the price?",
            "message_hour": 10,
            "message_length": 20,
            "high_intent_flag": 1,
            "prev_contacts": 0,
            "response_gap_hrs": 24.5,
            "missed_probability": 0.85,
            "predicted_missed": 1,
            "_customer_email": "priya@test.com",
            "_customer_name": "Priya",
            "_subject": "Pricing",
            "_received_time": "2025-01-15 10:00 UTC",
        }
        result = db.insert_lead(lead)
        assert result is True

    def test_get_lead(self, db):
        lead = {"lead_id": "L00001", "channel": "Email", "message_text": "Hello"}
        db.insert_lead(lead)
        result = db.get_lead("L00001")
        assert result is not None
        assert result["lead_id"] == "L00001"
        assert result["channel"] == "Email"

    def test_get_lead_not_found(self, db):
        result = db.get_lead("NONEXISTENT")
        assert result is None

    def test_lead_exists(self, db):
        db.insert_lead({"lead_id": "L00001"})
        assert db.lead_exists("L00001") is True
        assert db.lead_exists("L99999") is False

    def test_insert_leads_batch(self, db):
        leads = [
            {"lead_id": f"L{i:05d}", "channel": "Email"}
            for i in range(10)
        ]
        count = db.insert_leads_batch(leads)
        assert count == 10
        assert db.count_leads() == 10

    def test_get_leads_with_status_filter(self, db):
        db.insert_lead({"lead_id": "L00001", "predicted_missed": 1})
        db.insert_lead({"lead_id": "L00002", "predicted_missed": 0})
        missed = db.get_leads(status="missed")
        assert len(missed) == 1
        responded = db.get_leads(status="responded")
        assert len(responded) == 1

    def test_get_leads_with_channel_filter(self, db):
        db.insert_lead({"lead_id": "L00001", "channel": "Email"})
        db.insert_lead({"lead_id": "L00002", "channel": "WhatsApp"})
        email_leads = db.get_leads(channel="Email")
        assert len(email_leads) == 1

    def test_get_leads_with_limit(self, db):
        for i in range(20):
            db.insert_lead({"lead_id": f"L{i:05d}"})
        leads = db.get_leads(limit=5)
        assert len(leads) == 5

    def test_count_leads(self, db):
        db.insert_lead({"lead_id": "L00001", "predicted_missed": 1})
        db.insert_lead({"lead_id": "L00002", "predicted_missed": 0})
        assert db.count_leads() == 2
        assert db.count_leads("missed") == 1
        assert db.count_leads("responded") == 1


class TestReplies:
    """Test reply log operations."""

    def test_insert_reply(self, db):
        db.insert_lead({"lead_id": "L00001", "channel": "Email"})
        reply = {
            "lead_id": "L00001",
            "customer_email": "priya@test.com",
            "customer_name": "Priya",
            "reply_subject": "Re: Pricing",
            "reply_body": "Hello!",
            "detected_intent": "pricing",
            "replied_at": "2025-01-15 10:00 UTC",
            "channel": "Email",
        }
        reply_id = db.insert_reply(reply)
        assert reply_id > 0

    def test_get_replies(self, db):
        db.insert_lead({"lead_id": "L00001"})
        db.insert_lead({"lead_id": "L00002"})
        db.insert_reply({"lead_id": "L00001", "reply_subject": "Re: Hi"})
        db.insert_reply({"lead_id": "L00002", "reply_subject": "Re: Bye"})
        replies = db.get_replies()
        assert len(replies) == 2

    def test_count_replies(self, db):
        db.insert_lead({"lead_id": "L00001"})
        db.insert_lead({"lead_id": "L00002"})
        db.insert_reply({"lead_id": "L00001"})
        db.insert_reply({"lead_id": "L00002"})
        assert db.count_replies() == 2


class TestFollowupStatus:
    """Test follow-up status operations."""

    def test_upsert_followup(self, db):
        db.upsert_followup("L00001", {
            "customer_name": "Priya",
            "customer_email": "priya@test.com",
            "auto_replied": True,
            "auto_replied_at": "2025-01-15 10:00 UTC",
        })
        result = db.get_followup("L00001")
        assert result is not None
        assert result["customer_name"] == "Priya"
        assert result["auto_replied"] is True

    def test_get_followup_not_found(self, db):
        result = db.get_followup("NONEXISTENT")
        assert result is None

    def test_get_all_followups(self, db):
        db.upsert_followup("L00001", {"auto_replied": True})
        db.upsert_followup("L00002", {"auto_replied": False})
        all_followups = db.get_all_followups()
        assert len(all_followups) == 2
        assert "L00001" in all_followups
        assert "L00002" in all_followups

    def test_mark_human_followed_up(self, db):
        db.upsert_followup("L00001", {"auto_replied": True})
        db.mark_human_followed_up("L00001")
        result = db.get_followup("L00001")
        assert result["human_followed_up"] is True
        assert result["human_followed_up_at"] is not None

    def test_mark_overdue_notified(self, db):
        db.upsert_followup("L00001", {"auto_replied": True})
        db.mark_overdue_notified("L00001")
        result = db.get_followup("L00001")
        assert result["overdue_notified"] is True

    def test_get_pending_followups(self, db):
        db.upsert_followup("L00001", {"auto_replied": True, "human_followed_up": False})
        db.upsert_followup("L00002", {"auto_replied": True, "human_followed_up": True})
        pending = db.get_pending_followups()
        assert len(pending) == 1
        assert pending[0]["lead_id"] == "L00001"


class TestNotifications:
    """Test notification operations."""

    def test_insert_notification(self, db):
        notif_id = db.insert_notification(
            "new_lead", "New Lead", "Priya needs follow-up",
            customer_name="Priya", lead_id="L00001"
        )
        assert notif_id > 0

    def test_get_notifications(self, db):
        db.insert_notification("info", "Test1", "msg1")
        db.insert_notification("info", "Test2", "msg2")
        notifs = db.get_notifications()
        assert len(notifs) == 2

    def test_get_unread_notifications(self, db):
        db.insert_notification("info", "Read", "msg")
        db.insert_notification("info", "Unread", "msg")
        notifs = db.get_notifications(unread_only=True)
        assert len(notifs) == 2

    def test_mark_notification_read(self, db):
        notif_id = db.insert_notification("info", "Test", "msg")
        db.mark_notification_read(notif_id)
        notifs = db.get_notifications(unread_only=True)
        assert len(notifs) == 0

    def test_mark_all_notifications_read(self, db):
        db.insert_notification("info", "Test1", "msg")
        db.insert_notification("info", "Test2", "msg")
        db.mark_all_notifications_read()
        notifs = db.get_notifications(unread_only=True)
        assert len(notifs) == 0

    def test_count_unread_notifications(self, db):
        db.insert_notification("info", "Test", "msg")
        assert db.count_unread_notifications() == 1
        db.mark_all_notifications_read()
        assert db.count_unread_notifications() == 0


class TestSentLeads:
    """Test sent leads dedup."""

    def test_mark_sent(self, db):
        db.mark_sent("L00001")
        assert db.is_sent("L00001") is True
        assert db.is_sent("L00002") is False

    def test_get_sent_ids(self, db):
        db.mark_sent("L00001")
        db.mark_sent("L00002")
        sent_ids = db.get_sent_ids()
        assert "L00001" in sent_ids
        assert "L00002" in sent_ids

    def test_mark_sent_idempotent(self, db):
        db.mark_sent("L00001")
        db.mark_sent("L00001")  # Should not raise
        assert db.is_sent("L00001") is True


class TestFailedLeads:
    """Test failed leads tracking."""

    def test_insert_failed_lead(self, db):
        db.insert_failed_lead("L00001", {
            "customer_email": "test@test.com",
            "error": "SMTP error",
            "attempts": 3,
        })
        failed = db.get_failed_leads()
        assert "L00001" in failed
        assert failed["L00001"]["error"] == "SMTP error"


class TestScanLog:
    """Test scan logging."""

    def test_log_scan(self, db):
        db.log_scan(scanned=10, missed_detected=3, replied=2, skipped=1)
        assert db.get_scan_count() == 1

    def test_get_scan_count(self, db):
        db.log_scan(10, 3, 2, 1)
        db.log_scan(15, 5, 3, 2)
        assert db.get_scan_count() == 2


class TestSeenEmailIds:
    """Test seen email ID tracking."""

    def test_mark_email_seen(self, db):
        db.mark_email_seen("<test@mail.com>")
        assert db.is_email_seen("<test@mail.com>") is True
        assert db.is_email_seen("<other@mail.com>") is False

    def test_get_seen_ids(self, db):
        db.mark_email_seen("<test1@mail.com>")
        db.mark_email_seen("<test2@mail.com>")
        seen = db.get_seen_ids()
        assert "<test1@mail.com>" in seen
        assert "<test2@mail.com>" in seen


class TestStats:
    """Test statistics gathering."""

    def test_get_stats(self, db):
        db.insert_lead({"lead_id": "L00001", "predicted_missed": 1})
        db.insert_lead({"lead_id": "L00002", "predicted_missed": 0})
        db.insert_reply({"lead_id": "L00001"})
        db.upsert_followup("L00001", {"auto_replied": True})
        db.log_scan(2, 1, 1, 0)

        stats = db.get_stats()
        assert stats["total_leads"] == 2
        assert stats["missed_leads"] == 1
        assert stats["auto_replied"] == 1
        assert stats["scan_count"] == 1
