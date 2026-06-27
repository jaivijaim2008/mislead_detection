"""
inbox_monitor.py — Missed-Lead Detector
Continuously monitors Gmail inbox, scores new emails with ML model,
auto-generates human-like replies, sends them, and logs everything.

Run modes:
  python src/inbox_monitor.py              ← Run once (single scan)
  python src/inbox_monitor.py --loop 300   ← Run every 300 seconds (5 min)
  python src/inbox_monitor.py --dry-run    ← Scan + generate replies without sending
"""

import os, sys, json, time, pickle
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from email_reader       import fetch_customer_emails
from smart_reply_engine import generate_reply, detect_intent
from auto_followup      import send_followup, _load_sent, _save_sent
from notifications      import notify_new_lead, notify_auto_reply, notify_overdue, notify_low_recovery_rate
from config             import RECOVERY_RATE_THRESHOLD

BASE     = os.path.dirname(__file__)
MODELS   = os.path.join(BASE, "..", "models")
OUT      = os.path.join(BASE, "..", "outputs", "leads_scored.csv")
LOG_DIR  = os.path.join(BASE, "..", "logs")
REPLY_LOG = os.path.join(LOG_DIR, "auto_replies.json")
FOLLOWUP_LOG = os.path.join(LOG_DIR, "followup_status.json")

os.makedirs(LOG_DIR, exist_ok=True)


def _load_reply_log() -> list:
    if os.path.exists(REPLY_LOG):
        with open(REPLY_LOG) as f:
            return json.load(f)
    return []


def _save_reply_log(entries: list):
    with open(REPLY_LOG, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def _load_followup_status() -> dict:
    if os.path.exists(FOLLOWUP_LOG):
        with open(FOLLOWUP_LOG) as f:
            return json.load(f)
    return {}


def _save_followup_status(status: dict):
    with open(FOLLOWUP_LOG, "w") as f:
        json.dump(status, f, indent=2, default=str)


def load_artefacts():
    """Load trained ML models for scoring."""
    ensemble_path = os.path.join(MODELS, "ensemble.pkl")
    scaler_path   = os.path.join(MODELS, "scaler.pkl")

    if not os.path.exists(ensemble_path) or not os.path.exists(scaler_path):
        print("[monitor] WARNING: ML models not found. Scoring will be skipped.")
        return None, None

    with open(ensemble_path, "rb") as f:
        ensemble = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return ensemble, scaler


# Hardcoded channel encoding (must match training data)
CHANNEL_MAP = {"email": 0, "phone inquiry": 1, "website chat": 2, "whatsapp": 3}

def score_email(df: pd.DataFrame, ensemble, scaler) -> pd.DataFrame:
    """Score a single email DataFrame using the trained ensemble."""
    df["channel_enc"] = df["channel"].apply(
        lambda c: CHANNEL_MAP.get(str(c).lower(), 0))

    intent_words = ["price", "buy", "interested", "demo", "quote", "available"]
    df["intent_score"] = df["message_text"].apply(
        lambda t: sum(w in str(t).lower() for w in intent_words))
    df["is_business_hours"] = df["message_hour"].between(9, 18).astype(int)
    df["gap_bucket"] = pd.cut(
        df["response_gap_hrs"],
        bins=[0, 6, 12, 24, 9999],
        labels=[0, 1, 2, 3]
    ).astype(int)

    features = ["channel_enc", "message_length", "high_intent_flag", "prev_contacts",
                "response_gap_hrs", "intent_score", "is_business_hours",
                "gap_bucket", "message_hour"]
    X = scaler.transform(df[features])
    df["missed_probability"] = ensemble.predict_proba(X)[:, 1]
    df["predicted_missed"] = (df["missed_probability"] >= 0.5).astype(int)
    return df


def run_scan(dry_run: bool = False) -> dict:
    """
    Single scan: fetch new emails, score, auto-reply to missed leads, log everything.
    Returns summary dict.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*60}")
    print(f"  INBOX SCAN — {now}")
    print(f"{'='*60}")

    # 1. Fetch new emails
    df = fetch_customer_emails(max_emails=30, search_since_days=7)
    if df.empty:
        print("[monitor] No new emails found.")
        return {"scanned": 0, "scored": 0, "replied": 0, "skipped": 0}

    # 2. Score with ML model
    ensemble, scaler = load_artefacts()
    if ensemble and scaler:
        df = score_email(df, ensemble, scaler)
        print(f"[monitor] Scored {len(df)} emails with ML ensemble")
    else:
        # Fallback: use intent-based scoring
        df["missed_probability"] = df.apply(
            lambda r: min(0.95, 0.3 + 0.2 * (1 - r["high_intent_flag"]) +
                         0.1 * min(r["response_gap_hrs"] / 100, 0.5)), axis=1)
        df["predicted_missed"] = (df["missed_probability"] >= 0.5).astype(int)
        print(f"[monitor] Scored {len(df)} emails with intent-based fallback")

    # 3. Save scored data
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    # 4. Auto-reply to missed leads
    reply_log = _load_reply_log()
    sent_ids = _load_sent()
    followup_status = _load_followup_status()
    replied_count = 0
    skipped_count = 0

    missed = df[df["predicted_missed"] == 1]
    not_missed = df[df["predicted_missed"] == 0]

    for _, row in missed.iterrows():
        lead_id = row["lead_id"]

        # Skip if already replied
        if lead_id in sent_ids:
            skipped_count += 1
            continue

        # Generate smart reply
        reply = generate_reply(
            customer_name=row["_customer_name"],
            customer_email=row["_customer_email"],
            subject=row["_subject"],
            message_text=row["message_text"],
            channel=row.get("channel", "Email"),
        )

        # Build email payload — include smart reply content so auto_followup
        # sends the intent-aware, personalized reply instead of the generic template
        email_payload = {
            "lead_id": lead_id,
            "customer_email": row["_customer_email"],
            "customer_name": row["_customer_name"],
            "channel": row.get("channel", "Email"),
            "subject": row["_subject"],
            "original_message_id": row.get("_message_id", ""),
            "reply_subject": reply["reply_subject"],
            "reply_body": reply["reply_body"],
            "detected_intent": reply["detected_intent"],
        }

        if dry_run:
            print(f"\n[DRY RUN] Would auto-reply to {row['_customer_name']} "
                  f"<{row['_customer_email']}>")
            print(f"  Intent: {reply['detected_intent']}")
            print(f"  Subject: {reply['reply_subject']}")
        else:
            # Send the auto-reply (dedup handled by sent_ids check above)
            success = send_followup(email_payload)
            if success:
                replied_count += 1

                # Log the reply
                reply_entry = {
                    "lead_id": lead_id,
                    "customer_name": row["_customer_name"],
                    "customer_email": row["_customer_email"],
                    "subject": row["_subject"],
                    "reply_subject": reply["reply_subject"],
                    "detected_intent": reply["detected_intent"],
                    "missed_probability": float(row["missed_probability"]),
                    "replied_at": now,
                    "channel": row.get("channel", "Email"),
                }
                reply_log.append(reply_entry)

                # Track follow-up status
                followup_status[lead_id] = {
                    "customer_name": row["_customer_name"],
                    "customer_email": row["_customer_email"],
                    "auto_replied": True,
                    "auto_replied_at": now,
                    "human_followed_up": False,
                    "human_followed_up_at": None,
                    "overdue_notified": False,
                }

                # Send notifications
                notify_auto_reply(row["_customer_name"], row["_customer_email"],
                                  reply["detected_intent"])

                print(f"[monitor] AUTO-REPLIED → {row['_customer_name']} "
                      f"({reply['detected_intent']})")

        # Notify sales team about new missed lead
        notify_new_lead(row["_customer_name"], row["_customer_email"],
                        row["response_gap_hrs"], row["missed_probability"])

    # 5. Check for overdue leads (no human follow-up within 24h)
    for lead_id, status in followup_status.items():
        if status.get("auto_replied") and not status.get("human_followed_up"):
            if not status.get("overdue_notified"):
                replied_at = status.get("auto_replied_at", "")
                if replied_at:
                    try:
                        dt = datetime.strptime(replied_at, "%Y-%m-%d %H:%M UTC")
                        hours_since = (datetime.now(timezone.utc) -
                                       dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
                        if hours_since > 24:
                            notify_overdue(status["customer_name"],
                                          status["customer_email"], hours_since)
                            status["overdue_notified"] = True
                            print(f"[monitor] OVERDUE ALERT → {status['customer_name']} "
                                  f"({hours_since:.0f}h without human follow-up)")
                    except (ValueError, TypeError):
                        pass

    # 6. Save all logs
    _save_reply_log(reply_log)
    _save_sent(sent_ids)
    _save_followup_status(followup_status)

    summary = {
        "scanned": len(df),
        "scored": len(df),
        "missed_detected": len(missed),
        "replied": replied_count,
        "skipped": skipped_count,
        "not_missed": len(not_missed),
        "timestamp": now,
    }

    # 7. Check recovery rate and alert if below threshold
    # Use all-time followup_status for consistent rate calculation
    total_missed_all = len(followup_status)
    handled_count = sum(
        1 for s in followup_status.values()
        if isinstance(s, dict) and (s.get("auto_replied") or s.get("human_followed_up"))
    )
    recovery_pct = (handled_count / total_missed_all * 100) if total_missed_all > 0 else 100.0
    if total_missed_all > 0 and recovery_pct < RECOVERY_RATE_THRESHOLD:
        # Cooldown: only alert once per 6 hours to avoid spam
        alert_log = os.path.join(LOG_DIR, "recovery_rate_alert.json")
        last_alert = {}
        if os.path.exists(alert_log):
            try:
                with open(alert_log) as f:
                    last_alert = json.load(f)
            except Exception:
                pass
        last_ts = last_alert.get("timestamp", "")
        try:
            last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
            hours_since = (datetime.now(timezone.utc) -
                           last_dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_since = 999  # no previous alert

        if hours_since >= 6:
            notify_low_recovery_rate(
                current_rate=recovery_pct,
                threshold=RECOVERY_RATE_THRESHOLD,
                missed_total=total_missed_all,
                handled=handled_count,
            )
            with open(alert_log, "w") as f:
                json.dump({"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                           "rate": recovery_pct}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE")
    print(f"  Emails scanned: {summary['scanned']}")
    print(f"  Missed leads:   {summary['missed_detected']}")
    print(f"  Auto-replied:   {summary['replied']}")
    print(f"  Skipped:        {summary['skipped']}")
    print(f"{'='*60}\n")

    return summary


def run_loop(interval: int = 300, dry_run: bool = False):
    """Run continuous monitoring loop."""
    print(f"[monitor] Starting continuous monitoring (every {interval}s)")
    while True:
        try:
            run_scan(dry_run=dry_run)
        except Exception as e:
            print(f"[monitor] ERROR during scan: {e}")
        print(f"[monitor] Next scan in {interval}s...")
        time.sleep(interval)


if __name__ == "__main__":
    import traceback
    args = sys.argv[1:]
    dry_run = "--dry-run" in args

    try:
        if "--loop" in args:
            idx = args.index("--loop")
            interval = int(args[idx + 1]) if idx + 1 < len(args) else 300
            run_loop(interval=interval, dry_run=dry_run)
        else:
            run_scan(dry_run=dry_run)
    except Exception as e:
        print(f"\n[monitor] FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
