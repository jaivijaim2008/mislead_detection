"""
test_live_followup.py — Missed-Lead Detector
Tests sending a real follow-up email via SMTP to verify the auto_followup works.
This directly tests the SMTP integration.

Usage:
    python tools/test_live_followup.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auto_followup import process_missed_leads

if __name__ == "__main__":
    SMTP_USER = os.getenv("SMTP_USER", "")
    if not SMTP_USER:
        print("ERROR: SMTP_USER env var not set.")
        sys.exit(1)

    print("=" * 60)
    print("  TESTING LIVE SMTP FOLLOW-UP EMAIL")
    print("=" * 60)
    print(f"  Sending from: {SMTP_USER}")
    print("=" * 60)

    # Test lead - will send follow-up to the user's own inbox
    test_leads = [
        {
            "lead_id": "TEST-LIVE-001",
            "customer_email": SMTP_USER,
            "customer_name": "Priya",
            "channel": "Email",
            "subject": "[TEST-INQUIRY] Price Inquiry - Product A",
            "original_message_id": "",
        },
    ]

    print(f"\n  Sending test follow-up to {SMTP_USER}...")
    result = process_missed_leads(test_leads)
    print(f"\n  Result: Sent {result['sent']}, Skipped {result['skipped']}")

    if result["sent"] > 0:
        print(f"\n{'='*60}")
        print(f"  SUCCESS! Follow-up email sent to your inbox.")
        print(f"  Check your Gmail (jaivijai188@gmail.com) to see it.")
        print(f"{'='*60}")
    else:
        print(f"\n  No emails were sent (check DEMO_MODE in auto_followup.py)")
