"""
database.py — Missed-Lead Detector
SQLite storage backend for leads, replies, notifications, and follow-up status.
Replaces JSON file reads/writes with proper database operations.

Usage:
    from database import get_db
    db = get_db()
    db.insert_lead(...)
    db.get_leads(...)
"""
import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "missed_leads.db")

# ── Singleton DB instance ───────────────────────────────────

_db_instance = None


def get_db(db_path: str = None) -> "Database":
    """Get or create the singleton Database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path or DB_PATH)
    return _db_instance


def reset_db():
    """Reset the singleton (for testing)."""
    global _db_instance
    _db_instance = None


class Database:
    """SQLite storage backend for the Missed-Lead Detector."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS leads (
                    lead_id TEXT PRIMARY KEY,
                    channel TEXT,
                    message_text TEXT,
                    message_hour INTEGER,
                    message_length INTEGER,
                    high_intent_flag INTEGER,
                    prev_contacts INTEGER,
                    response_gap_hrs REAL,
                    missed_probability REAL,
                    predicted_missed INTEGER,
                    customer_email TEXT,
                    customer_name TEXT,
                    subject TEXT,
                    received_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lead_id TEXT,
                    customer_email TEXT,
                    customer_name TEXT,
                    reply_subject TEXT,
                    reply_body TEXT,
                    detected_intent TEXT,
                    missed_probability REAL,
                    sent_at TEXT,
                    channel TEXT,
                    status TEXT DEFAULT 'sent'
                );

                CREATE TABLE IF NOT EXISTS followup_status (
                    lead_id TEXT PRIMARY KEY,
                    customer_name TEXT,
                    customer_email TEXT,
                    auto_replied INTEGER DEFAULT 0,
                    auto_replied_at TEXT,
                    human_followed_up INTEGER DEFAULT 0,
                    human_followed_up_at TEXT,
                    overdue_notified INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT,
                    title TEXT,
                    message TEXT,
                    customer_name TEXT,
                    lead_id TEXT,
                    read INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sent_leads (
                    lead_id TEXT PRIMARY KEY,
                    sent_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS failed_leads (
                    lead_id TEXT PRIMARY KEY,
                    customer_email TEXT,
                    customer_name TEXT,
                    subject TEXT,
                    error TEXT,
                    attempts INTEGER,
                    failed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS scan_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scanned INTEGER,
                    missed_detected INTEGER,
                    replied INTEGER,
                    skipped INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS seen_email_ids (
                    message_id TEXT PRIMARY KEY,
                    seen_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_leads_predicted ON leads(predicted_missed);
                CREATE INDEX IF NOT EXISTS idx_replies_lead ON replies(lead_id);
                CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(read);
                CREATE INDEX IF NOT EXISTS idx_followup_auto ON followup_status(auto_replied);
                CREATE INDEX IF NOT EXISTS idx_followup_human ON followup_status(human_followed_up);
            """)

    # ── Leads ───────────────────────────────────────────────

    def insert_lead(self, lead: dict) -> bool:
        """Insert or replace a lead."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO leads
                (lead_id, channel, message_text, message_hour, message_length,
                 high_intent_flag, prev_contacts, response_gap_hrs,
                 missed_probability, predicted_missed,
                 customer_email, customer_name, subject, received_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lead.get("lead_id", ""),
                lead.get("channel", ""),
                lead.get("message_text", ""),
                lead.get("message_hour", 0),
                lead.get("message_length", 0),
                lead.get("high_intent_flag", 0),
                lead.get("prev_contacts", 0),
                lead.get("response_gap_hrs", 0),
                lead.get("missed_probability", 0),
                lead.get("predicted_missed", 0),
                lead.get("_customer_email") or lead.get("customer_email", ""),
                lead.get("_customer_name") or lead.get("customer_name", ""),
                lead.get("_subject") or lead.get("subject", ""),
                lead.get("_received_time") or lead.get("received_time", ""),
            ))
        return True

    def insert_leads_batch(self, leads: List[dict]) -> int:
        """Insert multiple leads. Returns count inserted."""
        count = 0
        for lead in leads:
            self.insert_lead(lead)
            count += 1
        return count

    def get_lead(self, lead_id: str) -> Optional[dict]:
        """Get a single lead by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM leads WHERE lead_id = ?", (lead_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_leads(self, status: str = None, channel: str = None,
                  high_intent: bool = None, limit: int = 50,
                  offset: int = 0) -> List[dict]:
        """Get leads with filtering."""
        query = "SELECT * FROM leads WHERE 1=1"
        params = []

        if status == "missed":
            query += " AND predicted_missed = 1"
        elif status == "responded":
            query += " AND predicted_missed = 0"

        if channel:
            query += " AND LOWER(channel) = LOWER(?)"
            params.append(channel)

        if high_intent is not None:
            query += " AND high_intent_flag = ?"
            params.append(1 if high_intent else 0)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def count_leads(self, status: str = None) -> int:
        """Count leads."""
        query = "SELECT COUNT(*) FROM leads"
        params = []
        if status == "missed":
            query += " WHERE predicted_missed = 1"
        elif status == "responded":
            query += " WHERE predicted_missed = 0"

        with self._conn() as conn:
            return conn.execute(query, params).fetchone()[0]

    def lead_exists(self, lead_id: str) -> bool:
        """Check if a lead exists."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM leads WHERE lead_id = ?", (lead_id,)
            ).fetchone()
            return row is not None

    # ── Replies ─────────────────────────────────────────────

    def insert_reply(self, reply: dict) -> int:
        """Insert a reply log entry. Returns the insert ID."""
        # Check for duplicate (lead_id + sent_at)
        lead_id = reply.get("lead_id", "")
        sent_at = reply.get("replied_at") or reply.get("sent_at", "")
        if lead_id and sent_at:
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT id FROM replies WHERE lead_id = ? AND sent_at = ?",
                    (lead_id, sent_at)
                ).fetchone()
                if existing:
                    return existing["id"]

        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO replies
                (lead_id, customer_email, customer_name, reply_subject, reply_body,
                 detected_intent, missed_probability, sent_at, channel)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lead_id,
                reply.get("customer_email", ""),
                reply.get("customer_name", ""),
                reply.get("reply_subject", ""),
                reply.get("reply_body", ""),
                reply.get("detected_intent", ""),
                reply.get("missed_probability", 0),
                sent_at,
                reply.get("channel", ""),
            ))
            return cursor.lastrowid

    def get_replies(self, limit: int = 100) -> List[dict]:
        """Get reply log entries."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM replies ORDER BY sent_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def count_replies(self) -> int:
        """Count total replies."""
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM replies").fetchone()[0]

    # ── Follow-up Status ────────────────────────────────────

    def upsert_followup(self, lead_id: str, data: dict):
        """Insert or update follow-up status."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO followup_status
                (lead_id, customer_name, customer_email,
                 auto_replied, auto_replied_at,
                 human_followed_up, human_followed_up_at,
                 overdue_notified, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lead_id,
                data.get("customer_name", ""),
                data.get("customer_email", ""),
                1 if data.get("auto_replied") else 0,
                data.get("auto_replied_at", ""),
                1 if data.get("human_followed_up") else 0,
                data.get("human_followed_up_at", ""),
                1 if data.get("overdue_notified") else 0,
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            ))

    def get_followup(self, lead_id: str) -> Optional[dict]:
        """Get follow-up status for a lead."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM followup_status WHERE lead_id = ?", (lead_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["auto_replied"] = bool(d["auto_replied"])
                d["human_followed_up"] = bool(d["human_followed_up"])
                d["overdue_notified"] = bool(d["overdue_notified"])
                return d
            return None

    def get_all_followups(self) -> dict:
        """Get all follow-up statuses as a dict keyed by lead_id."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM followup_status").fetchall()
            result = {}
            for row in rows:
                d = dict(row)
                lead_id = d.pop("lead_id")
                d["auto_replied"] = bool(d["auto_replied"])
                d["human_followed_up"] = bool(d["human_followed_up"])
                d["overdue_notified"] = bool(d["overdue_notified"])
                result[lead_id] = d
            return result

    def mark_human_followed_up(self, lead_id: str):
        """Mark a lead as human-followed-up."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE followup_status
                SET human_followed_up = 1,
                    human_followed_up_at = ?,
                    updated_at = ?
                WHERE lead_id = ?
            """, (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                lead_id,
            ))

    def mark_overdue_notified(self, lead_id: str):
        """Mark a lead as overdue-notified."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE followup_status
                SET overdue_notified = 1, updated_at = ?
                WHERE lead_id = ?
            """, (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                lead_id,
            ))

    def get_pending_followups(self) -> List[dict]:
        """Get leads that need human follow-up."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM followup_status
                WHERE auto_replied = 1 AND human_followed_up = 0
            """).fetchall()
            return [dict(row) for row in rows]

    # ── Notifications ───────────────────────────────────────

    def insert_notification(self, ntype: str, title: str, message: str,
                           customer_name: str = "", lead_id: str = "") -> int:
        """Insert a notification. Returns the insert ID."""
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO notifications (type, title, message, customer_name, lead_id)
                VALUES (?, ?, ?, ?, ?)
            """, (ntype, title, message, customer_name, lead_id))
            return cursor.lastrowid

    def get_notifications(self, unread_only: bool = False, limit: int = 50) -> List[dict]:
        """Get notifications."""
        query = "SELECT * FROM notifications"
        if unread_only:
            query += " WHERE read = 0"
        query += " ORDER BY created_at DESC LIMIT ?"

        with self._conn() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["read"] = bool(d["read"])
                result.append(d)
            return result

    def mark_notification_read(self, notification_id: int):
        """Mark a notification as read."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE notifications SET read = 1 WHERE id = ?",
                (notification_id,)
            )

    def mark_all_notifications_read(self):
        """Mark all notifications as read."""
        with self._conn() as conn:
            conn.execute("UPDATE notifications SET read = 1")

    def count_unread_notifications(self) -> int:
        """Count unread notifications."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM notifications WHERE read = 0"
            ).fetchone()[0]

    # ── Sent Leads (Dedup) ──────────────────────────────────

    def is_sent(self, lead_id: str) -> bool:
        """Check if a lead has already been sent a follow-up."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sent_leads WHERE lead_id = ?", (lead_id,)
            ).fetchone()
            return row is not None

    def mark_sent(self, lead_id: str):
        """Mark a lead as sent."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sent_leads (lead_id) VALUES (?)",
                (lead_id,)
            )

    def get_sent_ids(self) -> set:
        """Get all sent lead IDs."""
        with self._conn() as conn:
            rows = conn.execute("SELECT lead_id FROM sent_leads").fetchall()
            return {row["lead_id"] for row in rows}

    # ── Failed Leads ────────────────────────────────────────

    def insert_failed_lead(self, lead_id: str, data: dict):
        """Record a failed lead send."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO failed_leads
                (lead_id, customer_email, customer_name, subject, error, attempts, failed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                lead_id,
                data.get("customer_email", ""),
                data.get("customer_name", ""),
                data.get("subject", ""),
                data.get("error", ""),
                data.get("attempts", 0),
                data.get("failed_at", ""),
            ))

    def get_failed_leads(self) -> dict:
        """Get all failed leads."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM failed_leads").fetchall()
            return {row["lead_id"]: dict(row) for row in rows}

    # ── Scan Log ────────────────────────────────────────────

    def log_scan(self, scanned: int, missed_detected: int,
                 replied: int, skipped: int):
        """Log a scan operation."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO scan_log (scanned, missed_detected, replied, skipped)
                VALUES (?, ?, ?, ?)
            """, (scanned, missed_detected, replied, skipped))

    def get_scan_count(self) -> int:
        """Get total number of scans performed."""
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM scan_log").fetchone()[0]

    # ── Seen Email IDs ──────────────────────────────────────

    def is_email_seen(self, message_id: str) -> bool:
        """Check if an email has been seen."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM seen_email_ids WHERE message_id = ?",
                (message_id,)
            ).fetchone()
            return row is not None

    def mark_email_seen(self, message_id: str):
        """Mark an email as seen."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen_email_ids (message_id) VALUES (?)",
                (message_id,)
            )

    def get_seen_ids(self) -> set:
        """Get all seen email IDs."""
        with self._conn() as conn:
            rows = conn.execute("SELECT message_id FROM seen_email_ids").fetchall()
            return {row["message_id"] for row in rows}

    # ── Stats ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        total_leads = self.count_leads()
        missed_leads = self.count_leads("missed")
        auto_replied = self.count_replies()
        followups = self.get_all_followups()
        human_followed = sum(1 for f in followups.values() if f.get("human_followed_up"))
        pending = sum(1 for f in followups.values()
                     if f.get("auto_replied") and not f.get("human_followed_up"))
        recovery_rate = (len(followups) / missed_leads * 100) if missed_leads > 0 else 100.0

        return {
            "total_leads": total_leads,
            "missed_leads": missed_leads,
            "auto_replied": auto_replied,
            "human_followed_up": human_followed,
            "pending": pending,
            "recovery_rate": round(recovery_rate, 1),
            "scan_count": self.get_scan_count(),
        }

    # ── Utilities ───────────────────────────────────────────

    def close(self):
        """Close the database connection."""
        pass  # Connections are managed per-operation

    def __del__(self):
        self.close()
