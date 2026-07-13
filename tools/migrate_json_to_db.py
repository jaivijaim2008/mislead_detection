"""
migrate_json_to_db.py — Missed-Lead Detector
Migrates existing JSON log files into the SQLite database.

Usage:
    python tools/migrate_json_to_db.py
    python tools/migrate_json_to_db.py --dry-run    # Preview without writing
    python tools/migrate_json_to_db.py --reset      # Drop and recreate tables
"""
import os
import sys
import json
import argparse

# Ensure src/ is on path
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from database import Database, reset_db

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_json(path: str, default=None):
    """Load a JSON file."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load {path}: {e}")
    return default if default is not None else []


def migrate_sent_leads(db: Database, dry_run: bool = False):
    """Migrate sent_leads.json → sent_leads table."""
    path = os.path.join(LOG_DIR, "sent_leads.json")
    data = load_json(path, [])
    print(f"\n[Migrate] sent_leads.json: {len(data)} entries")

    if dry_run:
        print(f"  (dry run - would insert {len(data)} rows)")
        return

    for lead_id in data:
        db.mark_sent(lead_id)
    print(f"  [OK] Migrated {len(data)} sent lead IDs")


def migrate_failed_leads(db: Database, dry_run: bool = False):
    """Migrate failed_leads.json → failed_leads table."""
    path = os.path.join(LOG_DIR, "failed_leads.json")
    data = load_json(path, {})
    print(f"\n[Migrate] failed_leads.json: {len(data)} entries")

    if dry_run:
        print(f"  (dry run - would insert {len(data)} rows)")
        return

    for lead_id, info in data.items():
        db.insert_failed_lead(lead_id, info)
    print(f"  [OK] Migrated {len(data)} failed leads")


def migrate_auto_replies(db: Database, dry_run: bool = False):
    """Migrate auto_replies.json → replies table."""
    path = os.path.join(LOG_DIR, "auto_replies.json")
    data = load_json(path, [])
    print(f"\n[Migrate] auto_replies.json: {len(data)} entries")

    if dry_run:
        print(f"  (dry run - would insert {len(data)} rows)")
        return

    for reply in data:
        db.insert_reply(reply)
    print(f"  [OK] Migrated {len(data)} reply log entries")


def migrate_followup_status(db: Database, dry_run: bool = False):
    """Migrate followup_status.json → followup_status table."""
    path = os.path.join(LOG_DIR, "followup_status.json")
    data = load_json(path, {})
    print(f"\n[Migrate] followup_status.json: {len(data)} entries")

    if dry_run:
        print(f"  (dry run - would insert {len(data)} rows)")
        return

    for lead_id, status in data.items():
        db.upsert_followup(lead_id, status)
    print(f"  [OK] Migrated {len(data)} follow-up status entries")


def migrate_notifications(db: Database, dry_run: bool = False):
    """Migrate notifications.json → notifications table."""
    path = os.path.join(LOG_DIR, "notifications.json")
    data = load_json(path, [])
    print(f"\n[Migrate] notifications.json: {len(data)} entries")

    if dry_run:
        print(f"  (dry run - would insert {len(data)} rows)")
        return

    for notif in data:
        db.insert_notification(
            ntype=notif.get("type", "info"),
            title=notif.get("title", ""),
            message=notif.get("message", ""),
            customer_name=notif.get("customer_name", ""),
            lead_id=notif.get("lead_id", ""),
        )
        # Mark as read if it was read
        if notif.get("read"):
            # Get the last inserted ID
            with db._conn() as conn:
                row = conn.execute(
                    "SELECT id FROM notifications ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    db.mark_notification_read(row["id"])

    print(f"  [OK] Migrated {len(data)} notifications")


def migrate_seen_email_ids(db: Database, dry_run: bool = False):
    """Migrate seen_email_ids.json → seen_email_ids table."""
    path = os.path.join(LOG_DIR, "seen_email_ids.json")
    data = load_json(path, [])
    print(f"\n[Migrate] seen_email_ids.json: {len(data)} entries")

    if dry_run:
        print(f"  (dry run - would insert {len(data)} rows)")
        return

    for msg_id in data:
        db.mark_email_seen(msg_id)
    print(f"  [OK] Migrated {len(data)} seen email IDs")


def migrate_scored_leads(db: Database, dry_run: bool = False):
    """Migrate leads_scored.csv → leads table."""
    import pandas as pd
    path = os.path.join(BASE_DIR, "outputs", "leads_scored.csv")
    if not os.path.exists(path):
        print(f"\n[Migrate] leads_scored.csv: not found, skipping")
        return

    df = pd.read_csv(path)
    print(f"\n[Migrate] leads_scored.csv: {len(df)} rows")

    if dry_run:
        print(f"  (dry run - would insert {len(df)} rows)")
        return

    for _, row in df.iterrows():
        db.insert_lead(row.to_dict())

    print(f"  [OK] Migrated {len(df)} leads")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Migrate JSON logs to SQLite")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    print("=" * 60)
    print("  MISSED-LEAD DETECTOR — JSON → SQLite Migration")
    print("=" * 60)

    if args.reset:
        print("\n[Reset] Dropping existing database...")
        db_path = os.path.join(DATA_DIR, "missed_leads.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"  Deleted {db_path}")
        reset_db()

    db = Database()
    print(f"\n[DB] Database: {db.db_path}")

    # Run all migrations
    migrate_sent_leads(db, args.dry_run)
    migrate_failed_leads(db, args.dry_run)
    migrate_auto_replies(db, args.dry_run)
    migrate_followup_status(db, args.dry_run)
    migrate_notifications(db, args.dry_run)
    migrate_seen_email_ids(db, args.dry_run)
    migrate_scored_leads(db, args.dry_run)

    # Print summary
    print("\n" + "=" * 60)
    print("  MIGRATION SUMMARY")
    print("=" * 60)
    stats = db.get_stats()
    print(f"  Total leads:        {stats['total_leads']}")
    print(f"  Missed leads:       {stats['missed_leads']}")
    print(f"  Auto-replied:       {stats['auto_replied']}")
    print(f"  Human followed up:  {stats['human_followed_up']}")
    print(f"  Pending:            {stats['pending']}")
    print(f"  Recovery rate:      {stats['recovery_rate']}%")
    print(f"  Scan count:         {stats['scan_count']}")
    print("=" * 60)

    if args.dry_run:
        print("\n  (dry run — no changes made)")
    else:
        print("\n  [OK] Migration complete!")
        print(f"  Database: {db.db_path}")


if __name__ == "__main__":
    main()
