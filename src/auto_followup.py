"""
auto_followup.py — Missed-Lead Detector
Sends a threaded, context-aware follow-up email to a missed lead.
Duplicate-prevention: tracks sent IDs so the same lead never gets two emails.

-- REAL SMTP USAGE -------------------------------------------------
Set environment variables before running:
    SMTP_HOST   = smtp.gmail.com
    SMTP_PORT   = 587
    SMTP_USER   = youremail@gmail.com
    SMTP_PASS   = your_app_password   (Gmail App Password)
    SENDER_NAME = Your Company Name
In demo mode the email is printed to stdout; no real SMTP required.
--------------------------------------------------------------------
"""

import os, json, smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText

SMTP_HOST   = os.getenv("SMTP_HOST",   "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER",   "")
SMTP_PASS   = os.getenv("SMTP_PASS",   "")
SENDER_NAME = os.getenv("SENDER_NAME", "Sales Team")
DEMO_MODE   = not bool(SMTP_USER)

SENT_LOG    = os.path.join(os.path.dirname(__file__), "..", "logs", "sent_leads.json")
os.makedirs(os.path.dirname(SENT_LOG), exist_ok=True)

def _load_sent() -> set:
    if os.path.exists(SENT_LOG):
        with open(SENT_LOG) as f:
            return set(json.load(f))
    return set()

def _save_sent(sent: set):
    with open(SENT_LOG, "w") as f:
        json.dump(sorted(sent), f, indent=2)

def _build_email(lead: dict) -> MIMEMultipart:
    msg                  = MIMEMultipart("alternative")
    msg["Subject"]       = f"Re: {lead.get('subject', 'Your Inquiry')}"
    msg["From"]          = f"{SENDER_NAME} <{SMTP_USER or 'demo@example.com'}>"
    msg["To"]            = lead["customer_email"]
    msg["Date"]          = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")

    orig_mid = lead.get("original_message_id")
    if orig_mid:
        msg["In-Reply-To"] = orig_mid
        msg["References"]  = orig_mid

    plain = (
        f"Hi {lead.get('customer_name', 'there')},\n\n"
        f"Thank you for reaching out to us via {lead.get('channel', 'our platform')}.\n\n"
        f"We noticed we haven't had a chance to follow up with you yet - "
        f"we sincerely apologise for the delay!\n\n"
        f"We'd love to help you. Could we schedule a quick call or chat at your convenience? "
        f"Please reply to this email or reach us directly.\n\n"
        f"Warm regards,\n{SENDER_NAME}"
    )

    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#333;max-width:600px;margin:auto">
      <p>Hi <strong>{lead.get('customer_name', 'there')}</strong>,</p>
      <p>Thank you for reaching out via <em>{lead.get('channel', 'our platform')}</em>.</p>
      <p>We noticed we haven't followed up - we sincerely apologise for the delay!</p>
      <p>We'd love to help. Could we schedule a quick call or chat at your convenience?
         Just reply to this email or reach us directly.</p>
      <p style="margin-top:2em">Warm regards,<br>
         <strong>{SENDER_NAME}</strong></p>
      <hr style="border:none;border-top:1px solid #eee;margin-top:2em">
      <p style="font-size:11px;color:#999">
        This email was sent by the {SENDER_NAME}.
      </p>
    </body></html>"""

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))
    return msg

def send_followup(lead: dict, force: bool = False) -> bool:
    lead_id = lead.get("lead_id", "UNKNOWN")
    sent    = _load_sent()

    if lead_id in sent and not force:
        print(f"[auto_followup] SKIP  - {lead_id} already followed up.")
        return False

    msg = _build_email(lead)

    if DEMO_MODE:
        print(f"\n{'='*60}")
        print(f"[auto_followup] DEMO MODE - would send email:")
        print(f"  Lead ID  : {lead_id}")
        print(f"  To       : {lead['customer_email']}")
        print(f"  Subject  : {msg['Subject']}")
        print(f"  Channel  : {lead.get('channel')}")
        print(f"  Threaded : {'Yes' if lead.get('original_message_id') else 'No'}")
        print(f"{'='*60}\n")
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(msg["From"], lead["customer_email"], msg.as_string())
        print(f"[auto_followup] SENT  - {lead_id} -> {lead['customer_email']}")

    sent.add(lead_id)
    _save_sent(sent)
    return True

def process_missed_leads(leads: list) -> dict:
    sent_count = skipped_count = 0
    for lead in leads:
        ok = send_followup(lead)
        if ok: sent_count += 1
        else:  skipped_count += 1
    print(f"\n[auto_followup] Batch done - Sent: {sent_count} | Skipped: {skipped_count}")
    return {"sent": sent_count, "skipped": skipped_count}

if __name__ == "__main__":
    sample_leads = [
        {
            "lead_id"             : "L0042",
            "customer_email"      : "priya@example.com",
            "customer_name"       : "Priya",
            "channel"             : "WhatsApp",
            "subject"             : "Your Inquiry - Product Demo",
            "original_message_id" : "<abc123@mail.example.com>",
        },
        {
            "lead_id"             : "L0099",
            "customer_email"      : "ravi@example.com",
            "customer_name"       : "Ravi",
            "channel"             : "Email",
            "subject"             : "Your Inquiry - Pricing",
        },
    ]
    process_missed_leads(sample_leads)
    print("\n== Duplicate prevention test (same IDs again) ==")
    process_missed_leads(sample_leads)   # Both should print SKIP
