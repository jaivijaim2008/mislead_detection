"""
send_test_emails.py — Missed-Lead Detector
Sends realistic test customer inquiry emails to the user's Gmail inbox
so the full IMAP → ML scoring → SMTP reply pipeline can be tested end-to-end.

Usage:
    python tools/send_test_emails.py

Requires same env vars as the main pipeline:
    SMTP_USER, SMTP_PASS
"""

import os, sys, smtplib, time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# Realistic test customer inquiries
TEST_EMAILS = [
    {
        "customer_name": "Priya Sharma",
        "customer_email": "jaivijai188@gmail.com",  # Sent to user's inbox
        "subject": "[TEST-INQUIRY] Price Inquiry - Product A",
        "message": """Hi there,

I came across your product on your website and I'm very interested. Could you please let me know the pricing details for the premium package? I'm looking to purchase within this week.

Also, do you offer any discounts for bulk orders?

Thanks,
Priya Sharma"""
    },
    {
        "customer_name": "Rahul Verma",
        "customer_email": "jaivijai188@gmail.com",
        "subject": "[TEST-INQUIRY] Demo Request",
        "message": """Hello,

I'd like to schedule a demo of your service for my team of 5 people. We're evaluating solutions and yours looks promising.

Could you let me know available time slots for next week?

Best regards,
Rahul Verma"""
    },
    {
        "customer_name": "Ananya Reddy",
        "customer_email": "jaivijai188@gmail.com",
        "subject": "[TEST-INQUIRY] Interested in your service",
        "message": """Hi,

I'm interested in learning more about what you offer. A friend recommended your service. Can you share a brochure or catalog?

Also, is there a free trial available?

Thanks,
Ananya Reddy"""
    },
    {
        "customer_name": "Vikram Singh",
        "customer_email": "jaivijai188@gmail.com",
        "subject": "[TEST-INQUIRY] Quote needed for bulk order",
        "message": """Hi Sales Team,

We are looking to place a bulk order for our company. Can you please provide a quote for 50 units?

Please include shipping costs and delivery timeline.

Regards,
Vikram Singh"""
    },
]


def send_email(recipient: str, subject: str, body: str, sender_name: str = "Test Customer") -> bool:
    """Send a test email to the monitored inbox."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{SMTP_USER}>"
    msg["To"] = recipient
    msg["Date"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")

    plain_body = body
    html_body = body.replace("\n", "<br>")

    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(f"<html><body>{html_body}</body></html>", "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(msg["From"], recipient, msg.as_string())
        print(f"  [OK] Sent: '{subject}' -> {recipient}")
        return True
    except Exception as e:
        print(f"  [FAIL] '{subject}' -> {e}")
        return False


if __name__ == "__main__":
    if not SMTP_USER or not SMTP_PASS:
        print("ERROR: SMTP_USER and SMTP_PASS env vars not set.")
        print("Run: export SMTP_USER='your.email@gmail.com'")
        print("     export SMTP_PASS='your-app-password'")
        sys.exit(1)

    print("=" * 60)
    print("  SENDING TEST CUSTOMER INQUIRY EMAILS")
    print("=" * 60)
    print(f"  From   : {SMTP_USER}")
    print(f"  To     : jaivijai188@gmail.com")
    print(f"  Count  : {len(TEST_EMAILS)} emails")
    print("=" * 60)

    successful = 0
    for i, email in enumerate(TEST_EMAILS):
        print(f"\n--- Email #{i+1}: {email['subject']}")
        ok = send_email(
            recipient=email["customer_email"],
            subject=email["subject"],
            body=email["message"],
            sender_name=email["customer_name"],
        )
        if ok:
            successful += 1
        time.sleep(2)  # Small delay between sends

    print(f"\n{'='*60}")
    print(f"  Sent {successful}/{len(TEST_EMAILS)} test emails successfully!")
    print(f"  They will appear in your Gmail inbox shortly.")
    print(f"  Then run: python src/orchestrator.py --preview")
    print(f"  To send follow-ups: python src/orchestrator.py --live")
    print(f"{'='*60}")
