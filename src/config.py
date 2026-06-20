"""
config.py — Missed-Lead Detector
Centralized business configuration. Customize your templates, pricing,
courses, and contact info here — or override via environment variables.

No code changes needed. Just edit the values below or set env vars.
"""

import os

# ── Business Identity ──────────────────────────────────────
COMPANY_NAME = os.getenv("COMPANY_NAME", "Your Company Name")
SENDER_NAME  = os.getenv("SENDER_NAME", "Sales Team")
TEAM_PHONE   = os.getenv("TEAM_PHONE", "+91-XXXXXXXXXX")
TEAM_EMAIL   = os.getenv("TEAM_EMAIL", "sales@yourcompany.com")
WEBSITE_URL  = os.getenv("WEBSITE_URL", "https://www.yourcompany.com")

# ── Courses & Programs ─────────────────────────────────────
# Add or remove courses as needed. These appear in template responses.
COURSES = [
    {
        "name": "Data Science & AI",
        "duration": "6 months",
        "price": "65,000",
        "emi_start": "4,999",
        "highlight": "Includes Python, ML, Deep Learning, and capstone project",
    },
    {
        "name": "Full Stack Development",
        "duration": "5 months",
        "price": "55,000",
        "emi_start": "3,999",
        "highlight": "React, Node.js, databases, and deployment",
    },
    {
        "name": "Digital Marketing",
        "duration": "3 months",
        "price": "25,000",
        "emi_start": "2,499",
        "highlight": "SEO, SEM, Social Media, Analytics, and live projects",
    },
    {
        "name": "Business Analytics (MBA Prep)",
        "duration": "4 months",
        "price": "45,000",
        "emi_start": "3,499",
        "highlight": "Excel, SQL, Tableau, Power BI, and case studies",
    },
    {
        "name": "HR Certification",
        "duration": "3 months",
        "price": "22,000",
        "emi_start": "1,999",
        "highlight": "Recruitment, payroll, labor laws, and HR analytics",
    },
]

# ── Batch Schedule ─────────────────────────────────────────
BATCH_SCHEDULES = {
    "weekday": "Mon-Fri, 7:00 PM - 9:30 PM",
    "weekend": "Sat-Sun, 10:00 AM - 1:00 PM",
    "fast_track": "Custom schedule — ask us!",
}
NEXT_BATCH_DATE = "Monday, July 7th"
SEATS_REMAINING = "limited"

# ── Placement Info ─────────────────────────────────────────
PLACEMENT_RATE = "90%+"
COMPANY_PARTNERS = "200+"
PLACEMENT_HIGHLIGHTS = [
    "Resume building & portfolio review",
    "Mock interviews with industry experts",
    "Direct referrals to partner companies",
    "Lifetime placement support",
]

# ── Pricing ────────────────────────────────────────────────
DISCOUNT_INFO = "Early-bird discount: 10% off if you enroll this week"
SCHOLARSHIP_INFO = "Merit-based scholarships available — ask us for details"
EMI_INFO = "0% EMI options available through our financing partners"

# ── Auto-Reply Behavior ────────────────────────────────────
# How fast to respond (in seconds) — adds a realistic delay
MIN_REPLY_DELAY = 30    # minimum 30 seconds
MAX_REPLY_DELAY = 300   # maximum 5 minutes

# ── Follow-Up Thresholds ───────────────────────────────────
HOURS_BEFORE_OVERDUE = 24     # alert if no human follow-up in this many hours
HOURS_BEFORE_ESCALATION = 48  # escalate after this many hours

# ── Email Signature ────────────────────────────────────────
EMAIL_SIGNATURE = f"""
Best regards,
{SENDER_NAME}
{COMPANY_NAME}
Phone: {TEAM_PHONE}
Email: {TEAM_EMAIL}
Web: {WEBSITE_URL}
"""

# ── Complaint Response Settings ────────────────────────────
COMPLAINT_RESOLUTION_TIME = "2 hours"
COMPLAINT_ESCALATION_NOTE = "I've already escalated this to our team lead"
