"""
notifications.py — Missed-Lead Detector
Multi-channel notification system for the sales team:
  1. Email alerts (via SMTP) when new leads arrive
  2. Desktop popups (via tkinter) for immediate attention
  3. Dashboard notifications (written to JSON for Streamlit to display)

Environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS — for email notifications
  NOTIFY_EMAIL — comma-separated list of sales team emails to alert
  SENDER_NAME — display name for notification emails
"""

import os, json, threading, datetime

SMTP_HOST   = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER", "")
SMTP_PASS   = os.getenv("SMTP_PASS", "")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", SMTP_USER)
SENDER_NAME  = os.getenv("SENDER_NAME", "Missed-Lead Detector")

BASE     = os.path.dirname(__file__)
LOG_DIR  = os.path.join(BASE, "..", "logs")
NOTIF_LOG = os.path.join(LOG_DIR, "notifications.json")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Dashboard Notification Store ─────────────────────────────────────────

_dashboard_notifications: list = []


def _load_notifications() -> list:
    global _dashboard_notifications
    if os.path.exists(NOTIF_LOG):
        with open(NOTIF_LOG) as f:
            _dashboard_notifications = json.load(f)
    return _dashboard_notifications


def _save_notifications(notifs: list):
    with open(NOTIF_LOG, "w") as f:
        json.dump(notifs[-200:], f, indent=2, default=str)  # keep last 200


def _add_dashboard_notification(ntype: str, title: str, message: str,
                                 customer_name: str = "", lead_id: str = ""):
    """Add a notification visible in the Streamlit dashboard."""
    notifs = _load_notifications()
    entry = {
        "type": ntype,       # "new_lead", "auto_reply", "overdue", "info"
        "title": title,
        "message": message,
        "customer_name": customer_name,
        "lead_id": lead_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "read": False,
    }
    notifs.append(entry)
    _save_notifications(notifs)
    return entry


def get_unread_notifications(limit: int = 50) -> list:
    """Get unread notifications for the dashboard."""
    notifs = _load_notifications()
    unread = [n for n in notifs if not n.get("read", False)]
    return unread[-limit:]


def mark_notification_read(index: int):
    """Mark a notification as read."""
    notifs = _load_notifications()
    if 0 <= index < len(notifs):
        notifs[index]["read"] = True
        _save_notifications(notifs)


def mark_all_read():
    """Mark all notifications as read."""
    notifs = _load_notifications()
    for n in notifs:
        n["read"] = True
    _save_notifications(notifs)


# ── Email Notifications ──────────────────────────────────────────────────

def _send_email_notification(to: str, subject: str, body: str):
    """Send an email notification to the sales team."""
    if not SMTP_USER or not NOTIFY_EMAIL:
        return  # Demo mode — no SMTP configured

    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = to
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [to], msg.as_string())
    except Exception as e:
        print(f"[notifications] Email send failed: {e}")


# ── Desktop Popup ────────────────────────────────────────────────────────

def _show_desktop_popup(title: str, message: str):
    """Show a desktop popup notification (non-blocking, Windows)."""
    def _popup_thread():
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.after(10000, root.destroy)  # auto-close after 10s
            messagebox.showinfo(title=title, message=message, parent=root)
            root.destroy()
        except Exception:
            pass  # Headless environment — skip popup
    t = threading.Thread(target=_popup_thread, daemon=True)
    t.start()


# ── Public Notification Functions ────────────────────────────────────────

def notify_new_lead(customer_name: str, customer_email: str,
                    gap_hrs: float, missed_prob: float):
    """Notify sales team about a new missed lead detected."""
    title = f"New Missed Lead: {customer_name}"
    message = (f"A new missed lead was detected!\n\n"
               f"Customer: {customer_name} <{customer_email}>\n"
               f"Response Gap: {gap_hrs:.1f} hours\n"
               f"Missed Probability: {missed_prob:.0%}\n\n"
               f"An auto-reply has been sent. Please follow up personally.")

    # Dashboard notification
    _add_dashboard_notification("new_lead", title, message,
                                customer_name=customer_name)

    # Email notification
    _send_email_notification(
        NOTIFY_EMAIL,
        f"[Missed Lead] {customer_name} — needs follow-up",
        message,
    )

    print(f"[notifications] NEW LEAD alert sent for {customer_name}")


def notify_auto_reply(customer_name: str, customer_email: str,
                      intent: str):
    """Notify that an auto-reply was sent."""
    title = f"Auto-Reply Sent: {customer_name}"
    message = (f"An auto-reply was sent to {customer_name}.\n\n"
               f"Customer: {customer_name} <{customer_email}>\n"
               f"Detected Intent: {intent}\n\n"
               f"The reply appears as a manual response. "
               f"Please monitor for replies.")

    _add_dashboard_notification("auto_reply", title, message,
                                customer_name=customer_name)

    print(f"[notifications] AUTO-REPLY notification for {customer_name}")


def notify_overdue(customer_name: str, customer_email: str,
                   hours_since: float):
    """Notify that a lead hasn't received human follow-up."""
    title = f"OVERDUE: {customer_name} — no human follow-up!"
    message = (f"⚠️ {customer_name} has not received a human follow-up!\n\n"
               f"Customer: {customer_name} <{customer_email}>\n"
               f"Hours since auto-reply: {hours_since:.0f}h\n\n"
               f"Please follow up immediately!")

    _add_dashboard_notification("overdue", title, message,
                                customer_name=customer_name)

    # Email notification for overdue
    _send_email_notification(
        NOTIFY_EMAIL,
        f"[URGENT] Overdue follow-up: {customer_name}",
        message,
    )

    # Desktop popup for immediate attention
    _show_desktop_popup(title, message)

    print(f"[notifications] OVERDUE alert sent for {customer_name}")


def notify_low_recovery_rate(current_rate: float, threshold: float,
                             missed_total: int, handled: int):
    """Notify when recovery rate drops below the configured threshold."""
    title = f"Low Recovery Rate Alert: {current_rate:.0f}%"
    message = (f"The lead recovery rate has dropped below your configured threshold.\n\n"
               f"Current Recovery Rate: {current_rate:.1f}%\n"
               f"Threshold: {threshold:.0f}%\n"
               f"Missed Leads: {missed_total}\n"
               f"Handled (auto-reply + follow-up): {handled}\n\n"
               f"Action needed: Review pending leads and assign follow-ups to the sales team.")

    _add_dashboard_notification("overdue", title, message)

    # Email notification
    _send_email_notification(
        NOTIFY_EMAIL,
        f"[ALERT] Recovery rate dropped to {current_rate:.0f}% — below {threshold:.0f}% threshold",
        message,
    )

    # Desktop popup for immediate attention
    _show_desktop_popup(title, message)

    print(f"[notifications] LOW RECOVERY RATE alert sent ({current_rate:.1f}% < {threshold:.0f}%)")


def notify_info(title: str, message: str):
    """Generic info notification."""
    _add_dashboard_notification("info", title, message)


if __name__ == "__main__":
    # Demo
    notify_new_lead("Priya Sharma", "priya@gmail.com", 48.5, 0.87)
    notify_auto_reply("Priya Sharma", "priya@gmail.com", "pricing")
    notify_overdue("Rahul Patel", "rahul@outlook.com", 36.0)
    print(f"\nUnread notifications: {len(get_unread_notifications())}")
    for n in get_unread_notifications():
        print(f"  [{n['type']}] {n['title']}")
