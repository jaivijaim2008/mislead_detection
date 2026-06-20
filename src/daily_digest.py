"""
daily_digest.py — Missed-Lead Detector
Compiles all inbox scans from the past 24 hours into a single email report.

Usage:
  python src/daily_digest.py              ← Send digest now
  python src/daily_digest.py --dry-run    ← Print digest to stdout, don't send

Environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS — for sending the digest
  NOTIFY_EMAIL  — recipient of the digest
  SENDER_NAME   — display name
  COMPANY_NAME  — company name in header
"""

import os, sys, json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Config ───────────────────────────────────────────────────────────────

SMTP_HOST   = os.getenv("SMTP_HOST",   "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER",   "")
SMTP_PASS   = os.getenv("SMTP_PASS",   "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", SMTP_USER)
SENDER_NAME  = os.getenv("SENDER_NAME", "Missed-Lead Detector")
COMPANY_NAME = os.getenv("COMPANY_NAME", "Your Company")

BASE     = os.path.dirname(__file__)
LOG_DIR  = os.path.join(BASE, "..", "logs")
OUT_DIR  = os.path.join(BASE, "..", "outputs")

AUTO_REPLIES_LOG  = os.path.join(LOG_DIR, "auto_replies.json")
NOTIFICATIONS_LOG = os.path.join(LOG_DIR, "notifications.json")
FOLLOWUP_LOG      = os.path.join(LOG_DIR, "followup_status.json")
SCORED_CSV        = os.path.join(OUT_DIR, "leads_scored.csv")


# ── Data Loading ─────────────────────────────────────────────────────────

def _load_json(path: str, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default if default is not None else []


def _filter_last_24h(entries: list, timestamp_key: str = "timestamp") -> list:
    """Keep only entries from the last 24 hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    result = []
    for entry in entries:
        ts = entry.get(timestamp_key, "")
        if not ts:
            continue
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                result.append(entry)
        except (ValueError, TypeError):
            pass
    return result


def _parse_csv_recent(csv_path: str, hours: int = 24) -> list:
    """Load leads_scored.csv and return rows from last N hours."""
    if not os.path.exists(csv_path):
        return []
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "_received_time" in df.columns:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            df["_received_time"] = pd.to_datetime(df["_received_time"], utc=True, errors="coerce")
            recent = df[df["_received_time"] >= cutoff]
            return recent.to_dict("records")
        return df.to_dict("records")
    except Exception:
        return []


# ── Digest Builder ───────────────────────────────────────────────────────

def build_digest() -> dict:
    """Compile all data from the last 24 hours into a summary dict."""
    now = datetime.now(timezone.utc)

    # Load all data
    auto_replies   = _load_json(AUTO_REPLIES_LOG, [])
    notifications  = _load_json(NOTIFICATIONS_LOG, [])
    followup       = _load_json(FOLLOWUP_LOG, {})
    scored_rows    = _parse_csv_recent(SCORED_CSV, hours=24)

    # Filter to last 24h
    recent_notifs = _filter_last_24h(notifications)

    # Compute stats
    new_leads     = [n for n in recent_notifs if n.get("type") == "new_lead"]
    overdue       = [n for n in recent_notifs if n.get("type") == "overdue"]

    # Follow-up stats
    total_tracked   = len(followup)
    auto_replied    = sum(1 for v in followup.values() if v.get("auto_replied"))
    human_followed  = sum(1 for v in followup.values() if v.get("human_followed_up"))
    still_pending   = sum(1 for v in followup.values()
                          if v.get("auto_replied") and not v.get("human_followed_up"))

    # Intent breakdown
    intent_counts = {}
    for entry in auto_replies:
        intent = entry.get("intent", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    # Average missed probability from scored data
    avg_missed_prob = 0.0
    if scored_rows:
        probs = [r.get("missed_probability", 0) for r in scored_rows]
        avg_missed_prob = sum(probs) / len(probs) if probs else 0.0

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "date": now.strftime("%B %d, %Y"),
        "emails_scored_24h": len(scored_rows),
        "new_leads": len(new_leads),
        "auto_replies_sent": len([n for n in recent_notifs if n.get("type") == "auto_reply"]),
        "overdue_alerts": len(overdue),
        "total_tracked_leads": total_tracked,
        "total_auto_replied": auto_replied,
        "total_human_followed": human_followed,
        "still_pending": still_pending,
        "follow_up_rate": (human_followed / auto_replied * 100) if auto_replied > 0 else 0,
        "intent_breakdown": intent_counts,
        "avg_missed_probability": avg_missed_prob,
        "new_leads_details": [{"name": n.get("customer_name", "—"), "title": n.get("title", "")}
                              for n in new_leads[:10]],
        "overdue_details": [{"name": n.get("customer_name", "—"), "title": n.get("title", "")}
                            for n in overdue[:10]],
    }


# ── HTML Email Template ──────────────────────────────────────────────────

def _build_html(d: dict) -> str:
    """Generate a beautiful HTML email digest."""
    intent_rows = ""
    for intent, count in sorted(d["intent_breakdown"].items(), key=lambda x: -x[1]):
        intent_rows += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #f0f0f5;font-size:14px;">{intent.title()}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0f0f5;font-size:14px;text-align:right;font-weight:600;">{count}</td>
        </tr>"""

    new_leads_html = ""
    for lead in d["new_leads_details"]:
        new_leads_html += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #f0f0f5;font-size:14px;">{lead['name']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0f0f5;font-size:13px;color:#666;">{lead['title']}</td>
        </tr>"""

    overdue_html = ""
    for lead in d["overdue_details"]:
        overdue_html += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #f0f0f5;font-size:14px;">{lead['name']}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0f0f5;font-size:13px;color:#e53e3e;">{lead['title']}</td>
        </tr>"""

    new_leads_section = ""
    if d["new_leads_details"]:
        new_leads_section = f"""
        <div style="margin-top:28px;">
          <h3 style="font-size:15px;color:#333;margin-bottom:12px;">New Leads (Last 24h)</h3>
          <table style="width:100%;border-collapse:collapse;background:#fafafa;border-radius:8px;overflow:hidden;">
            <tr style="background:#f0f0f5;">
              <th style="padding:8px 12px;text-align:left;font-size:12px;color:#666;text-transform:uppercase;letter-spacing:0.5px;">Customer</th>
              <th style="padding:8px 12px;text-align:left;font-size:12px;color:#666;text-transform:uppercase;letter-spacing:0.5px;">Details</th>
            </tr>
            {new_leads_html}
          </table>
        </div>"""

    overdue_section = ""
    if d["overdue_details"]:
        overdue_section = f"""
        <div style="margin-top:28px;">
          <h3 style="font-size:15px;color:#e53e3e;margin-bottom:12px;">⚠ Overdue — Needs Human Follow-Up</h3>
          <table style="width:100%;border-collapse:collapse;background:#fff5f5;border-radius:8px;overflow:hidden;border:1px solid #fed7d7;">
            <tr style="background:#fed7d7;">
              <th style="padding:8px 12px;text-align:left;font-size:12px;color:#9b2c2c;text-transform:uppercase;letter-spacing:0.5px;">Customer</th>
              <th style="padding:8px 12px;text-align:left;font-size:12px;color:#9b2c2c;text-transform:uppercase;letter-spacing:0.5px;">Alert</th>
            </tr>
            {overdue_html}
          </table>
        </div>"""

    intent_section = ""
    if d["intent_breakdown"]:
        intent_section = f"""
        <div style="margin-top:28px;">
          <h3 style="font-size:15px;color:#333;margin-bottom:12px;">Top Intents (All Time)</h3>
          <table style="width:100%;border-collapse:collapse;background:#fafafa;border-radius:8px;overflow:hidden;">
            <tr style="background:#f0f0f5;">
              <th style="padding:8px 12px;text-align:left;font-size:12px;color:#666;text-transform:uppercase;letter-spacing:0.5px;">Intent</th>
              <th style="padding:8px 12px;text-align:right;font-size:12px;color:#666;text-transform:uppercase;letter-spacing:0.5px;">Count</th>
            </tr>
            {intent_rows}
          </table>
        </div>"""

    follow_up_bar_color = "#48bb78" if d["follow_up_rate"] >= 50 else "#ecc94b" if d["follow_up_rate"] >= 25 else "#e53e3e"

    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin:0;padding:0;background:#f7f7fa;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
      <div style="max-width:600px;margin:0 auto;padding:24px;">

        <!-- Header -->
        <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:12px;padding:28px 32px;margin-bottom:20px;">
          <h1 style="margin:0;color:#fff;font-size:22px;font-weight:700;letter-spacing:-0.3px;">Daily Inbox Digest</h1>
          <p style="margin:6px 0 0;color:#a0aec0;font-size:13px;">{COMPANY_NAME} · {d['date']}</p>
        </div>

        <!-- Key Metrics -->
        <div style="background:#fff;border-radius:12px;padding:24px 28px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="display:flex;justify-content:space-between;flex-wrap:wrap;">
            <div style="text-align:center;flex:1;min-width:120px;padding:8px 0;">
              <div style="font-size:28px;font-weight:700;color:#2d3748;">{d['emails_scored_24h']}</div>
              <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Emails Scored</div>
            </div>
            <div style="text-align:center;flex:1;min-width:120px;padding:8px 0;border-left:1px solid #f0f0f5;border-right:1px solid #f0f0f5;">
              <div style="font-size:28px;font-weight:700;color:#e53e3e;">{d['new_leads']}</div>
              <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Missed Leads</div>
            </div>
            <div style="text-align:center;flex:1;min-width:120px;padding:8px 0;">
              <div style="font-size:28px;font-weight:700;color:#48bb78;">{d['auto_replies_sent']}</div>
              <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Auto-Replied</div>
            </div>
          </div>
        </div>

        <!-- Follow-Up Progress -->
        <div style="background:#fff;border-radius:12px;padding:24px 28px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <h3 style="font-size:15px;color:#333;margin:0 0 16px;">Follow-Up Progress</h3>
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:13px;color:#718096;">Human Follow-Up Rate</span>
            <span style="font-size:13px;font-weight:700;color:#2d3748;">{d['follow_up_rate']:.0f}%</span>
          </div>
          <div style="background:#edf2f7;border-radius:6px;height:8px;overflow:hidden;">
            <div style="background:{follow_up_bar_color};width:{d['follow_up_rate']:.0f}%;height:100%;border-radius:6px;transition:width 0.3s;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:12px;font-size:13px;color:#718096;">
            <span>Auto-replied: {d['total_auto_replied']}</span>
            <span>Human followed up: {d['total_human_followed']}</span>
            <span style="color:#e53e3e;">Pending: {d['still_pending']}</span>
          </div>
          {f'<div style="margin-top:12px;padding:10px 14px;background:#fff5f5;border-radius:8px;border:1px solid #fed7d7;font-size:13px;color:#9b2c2c;">{d["overdue_alerts"]} lead{"s" if d["overdue_alerts"] != 1 else ""} overdue — needs immediate attention</div>' if d['overdue_alerts'] > 0 else ''}
        </div>

        <!-- New Leads Table -->
        {new_leads_section}

        <!-- Overdue Table -->
        {overdue_section}

        <!-- Intent Breakdown -->
        {intent_section}

        <!-- Footer -->
        <div style="margin-top:28px;padding-top:16px;border-top:1px solid #e2e8f0;text-align:center;">
          <p style="font-size:12px;color:#a0aec0;margin:0;">
            Generated by {SENDER_NAME} · {d['generated_at']}<br>
            Avg. missed probability: {d['avg_missed_probability']:.1%}
          </p>
        </div>

      </div>
    </body>
    </html>"""


def _build_plain(d: dict) -> str:
    """Generate a plain-text fallback for the digest."""
    lines = [
        f"DAILY INBOX DIGEST — {d['date']}",
        f"Generated: {d['generated_at']}",
        "",
        "KEY METRICS",
        f"  Emails scored:    {d['emails_scored_24h']}",
        f"  Missed leads:     {d['new_leads']}",
        f"  Auto-replied:     {d['auto_replies_sent']}",
        f"  Overdue alerts:   {d['overdue_alerts']}",
        "",
        "FOLLOW-UP PROGRESS",
        f"  Rate:             {d['follow_up_rate']:.0f}%",
        f"  Auto-replied:     {d['total_auto_replied']}",
        f"  Human followed:   {d['total_human_followed']}",
        f"  Still pending:    {d['still_pending']}",
    ]

    if d["intent_breakdown"]:
        lines.append("")
        lines.append("TOP INTENTS (ALL TIME)")
        for intent, count in sorted(d["intent_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"  {intent.title():<20} {count}")

    if d["new_leads_details"]:
        lines.append("")
        lines.append("NEW LEADS (Last 24h)")
        for lead in d["new_leads_details"]:
            lines.append(f"  {lead['name']:<25} {lead['title']}")

    if d["overdue_details"]:
        lines.append("")
        lines.append("OVERDUE — NEEDS FOLLOW-UP")
        for lead in d["overdue_details"]:
            lines.append(f"  {lead['name']:<25} {lead['title']}")

    lines.append("")
    lines.append(f"Avg. missed probability: {d['avg_missed_probability']:.1%}")
    lines.append(f"— {SENDER_NAME}")
    return "\n".join(lines)


# ── Send ─────────────────────────────────────────────────────────────────

def send_digest(dry_run: bool = False) -> bool:
    """Build and send the daily digest email."""
    d = build_digest()

    subject = f"Daily Inbox Digest — {d['date']} · {d['new_leads']} missed leads"
    html = _build_html(d)
    plain = _build_plain(d)

    if dry_run:
        print(plain)
        print(f"\n[Daily Digest] DRY RUN — would send to {NOTIFY_EMAIL}")
        return True

    if not SMTP_USER or not NOTIFY_EMAIL:
        print("[Daily Digest] SMTP not configured — printing to stdout instead")
        print(plain)
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = NOTIFY_EMAIL

        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [NOTIFY_EMAIL], msg.as_string())

        print(f"[Daily Digest] Sent to {NOTIFY_EMAIL}")
        return True

    except Exception as e:
        print(f"[Daily Digest] Send failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    args = sys.argv[1:]
    dry_run = "--dry-run" in args

    try:
        send_digest(dry_run=dry_run)
    except Exception as e:
        print(f"\n[Daily Digest] FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
