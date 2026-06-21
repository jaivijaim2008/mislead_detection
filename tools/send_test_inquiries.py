"""
send_test_inquiries.py — Sends realistic customer inquiry emails
to the missedlead.detector@gmail.com inbox for testing the live scan.

These emails simulate different customer intents:
  - Price inquiry
  - Demo request
  - Course availability
  - Scholarship/EMI question
  - Urgent buy intent

Run:  python tools/send_test_inquiries.py
"""

import smtplib
import os
import time
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# ── Config ─────────────────────────────────────────────────
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# ── Realistic Test Emails ──────────────────────────────────
TEST_INQUIRIES = [
    {
        "from_name": "Rahul Sharma",
        "from_email": "rahul.sharma.test@gmail.com",
        "subject": "Interested in Data Science Course — Need Pricing",
        "body": """Hi,

I came across your Data Science course on your website and I'm very interested.
Could you please share the complete fee structure? I want to know:

1. Total course fee
2. Is there any early bird discount available?
3. Do you offer EMI options?

I'm currently working as a junior analyst and want to upskill.
Looking forward to hearing from you.

Best regards,
Rahul Sharma""",
        "intent": "price_inquiry"
    },
    {
        "from_name": "Priya Patel",
        "from_email": "priya.patel.test@gmail.com",
        "subject": "Can I get a demo class for ML course?",
        "body": """Hello Team,

I wanted to ask if I can attend a free demo class before enrolling.
I'm interested in your Machine Learning with Python course.

Also, do you provide placement assistance after course completion?

Thanks,
Priya Patel
+91-9876543210""",
        "intent": "demo_request"
    },
    {
        "from_name": "Amit Kumar",
        "from_email": "amit.kumar.test@gmail.com",
        "subject": "Is the next batch starting in July?",
        "body": """Hi,

I want to enroll in the Full Stack Development course.
When is the next batch starting? Is there a July batch available?

Please let me know the batch timings and schedule.
I can pay the full fee upfront if a seat is available.

Regards,
Amit Kumar""",
        "intent": "availability_check"
    },
    {
        "from_name": "Sneha Reddy",
        "from_email": "sneha.reddy.test@gmail.com",
        "subject": "Scholarship available for women?",
        "body": """Dear Sir/Madam,

I'm a final year B.Tech student. I'm interested in your AI & Deep Learning course.

Do you offer any scholarships for women students?
Also, can I pay in installments (EMI)?

I saw on your website that there's a 20% discount — is that still valid?

Thank you,
Sneha Reddy""",
        "intent": "scholarship_inquiry"
    },
    {
        "from_name": "Vikram Singh",
        "from_email": "vikram.singh.test@gmail.com",
        "subject": "URGENT — Want to enroll today!",
        "body": """Hello!

I spoke to your sales team yesterday about the Data Analytics course.
I've made my decision — I want to enroll RIGHT NOW.

Please send me the payment link immediately.
I want to start from the next batch itself.

My number: +91-9999988888
Call me ASAP.

Thanks,
Vikram Singh""",
        "intent": "buy_intent"
    },
    {
        "from_name": "Ananya Iyer",
        "from_email": "ananya.iyer.test@gmail.com",
        "subject": "Course comparison — Data Science vs AI",
        "body": """Hi,

I'm confused between your Data Science course and AI & Deep Learning course.
Can you help me understand:

1. Which one has better placement support?
2. What's the difference in curriculum?
3. Which course would you recommend for someone with 2 years of experience?

Please share the course brochures.

Best,
Ananya Iyer""",
        "intent": "comparison_inquiry"
    },
    {
        "from_name": "Karthik Menon",
        "from_email": "karthik.menon.test@gmail.com",
        "subject": "Do you offer corporate training?",
        "body": """Dear Team,

I'm the HR Manager at TechCorp Solutions. We want to upskill our team of 15 developers in Cloud Computing and DevOps.

Do you offer corporate/bulk training packages?
What would be the cost for 15 people?

Please share the details at your earliest convenience.

 Regards,
Karthik Menon
HR Manager, TechCorp Solutions""",
        "intent": "corporate_inquiry"
    },
    {
        "from_name": "Deepa Nair",
        "from_email": "deepa.nair.test@gmail.com",
        "subject": "Interested but need more time",
        "body": """Hello,

I attended your webinar last week on Python for Data Science.
The content was really good.

However, I need some more time to decide.
Can you please follow up with me next week?
I might need a payment plan option.

Thanks,
Deepa Nair""",
        "intent": "follow_up_needed"
    }
]


def create_email(inquiry: dict) -> MIMEMultipart:
    """Create a MIME email from an inquiry dict."""
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{inquiry['from_name']} <{inquiry['from_email']}>"
    msg["To"] = SMTP_USER
    msg["Subject"] = inquiry["subject"]
    msg["Reply-To"] = inquiry["from_email"]

    # Add a unique Message-ID-like header to look realistic
    msg["X-Mailer"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    # Plain text body
    msg.attach(MIMEText(inquiry["body"], "plain", "utf-8"))

    # HTML body (looks more realistic)
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; font-size: 14px; color: #333;">
    <p>{inquiry['body'].replace(chr(10), '<br>')}</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    return msg


def send_test_emails():
    """Send all test inquiry emails with delays to simulate real-world timing."""
    print("=" * 60)
    print("  Sending Test Customer Inquiry Emails")
    print(f"  Target Inbox: {SMTP_USER}")
    print(f"  Total Emails: {len(TEST_INQUIRIES)}")
    print("=" * 60)
    print()

    if not SMTP_USER or not SMTP_PASS:
        print("[ERROR] Set SMTP_USER and SMTP_PASS environment variables first.")
        print("  Example:  export SMTP_USER=your@gmail.com")
        print("           export SMTP_PASS=xxxx-xxxx-xxxx-xxxx")
        return

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        print("[OK] Connected to Gmail SMTP server\n")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("   Make sure SMTP_USER and SMTP_PASS are correct.")
        return

    sent_count = 0
    for i, inquiry in enumerate(TEST_INQUIRIES, 1):
        try:
            msg = create_email(inquiry)

            # Send the email
            server.sendmail(SMTP_USER, [SMTP_USER], msg.as_string())
            sent_count += 1

            print(f"  [{i}/{len(TEST_INQUIRIES)}] [SENT] {inquiry['subject']}")
            print(f"              From: {inquiry['from_name']} <{inquiry['from_email']}>")
            print(f"              Intent: {inquiry['intent']}")
            print()

            # Random delay between emails (2-5 seconds) to look natural
            if i < len(TEST_INQUIRIES):
                delay = random.uniform(2.0, 5.0)
                time.sleep(delay)

        except Exception as e:
            print(f"  [{i}/{len(TEST_INQUIRIES)}] [FAIL] {inquiry['subject']}")
            print(f"              Error: {e}")
            print()

    server.quit()

    print("=" * 60)
    print(f"  [DONE] Sent {sent_count}/{len(TEST_INQUIRIES)} test emails")
    print(f"  Check your inbox at {SMTP_USER}")
    print()
    print("  Next steps:")
    print("  1. Go to your Streamlit dashboard")
    print("  2. Click 'Trigger Scan Now' on Command Center")
    print("  3. Watch the ML model score these leads!")
    print("=" * 60)


if __name__ == "__main__":
    send_test_emails()
