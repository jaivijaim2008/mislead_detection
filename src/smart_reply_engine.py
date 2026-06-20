"""
smart_reply_engine.py — Missed-Lead Detector
Generates human-like, context-aware reply emails based on ML intent classification.
No LLM API required — uses template selection + personalization + varied phrasing.

The client should NOT be able to tell this is automated.
"""

import os, re, random
from datetime import datetime

SENDER_NAME = os.getenv("SENDER_NAME", "Sales Team")
COMPANY_NAME = os.getenv("COMPANY_NAME", "Our Company")
TEAM_PHONE = os.getenv("TEAM_PHONE", "+91-XXXXXXXXXX")

# ── Intent Detection ─────────────────────────────────────────────────────

PRICING_KEYWORDS = ["price", "pricing", "cost", "fee", "fees", "how much", "quote",
                     "budget", "affordable", "expensive", "discount", "emi", "payment"]
DEMO_KEYWORDS = ["demo", "trial", "sample", "try", "test", "see it", "walkthrough",
                  "preview", "demonstration", "show me"]
COURSE_KEYWORDS = ["course", "program", "batch", "class", "training", "certification",
                    "curriculum", "syllabus", "module", "mba", "diploma"]
PLACEMENT_KEYWORDS = ["placement", "job", "career", "hiring", "employment",
                       "opportunity", "internship", "resume"]
COMPLAINT_KEYWORDS = ["complaint", "issue", "problem", "not working", "unhappy",
                       "dissatisfied", "frustrated", "bad", "terrible", "worst"]
INTEREST_KEYWORDS = ["interested", "want to", "looking for", "need", "enquire",
                      "information", "details", "tell me", "share", "help"]
AVAILABILITY_KEYWORDS = ["available", "timing", "schedule", "when", "start",
                          "next batch", "date", "duration", "weekend", "weekday"]
URGENT_KEYWORDS = ["urgent", "asap", "immediately", "today", "tomorrow", "hurry",
                    "quick", "fast", "emergency", "critical"]

GREETINGS = ["Hi", "Hello", "Hey", "Dear"]
TIME_GREETINGS = {
    "morning": ["Good morning", "Hi", "Hello"],
    "afternoon": ["Good afternoon", "Hi", "Hello"],
    "evening": ["Good evening", "Hi", "Hello"],
}

def _time_of_day() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    return "evening"

def detect_intent(message_text: str) -> dict:
    """Classify email intent. Returns dict with intent scores."""
    text = message_text.lower()
    intents = {
        "pricing": sum(1 for w in PRICING_KEYWORDS if w in text),
        "demo": sum(1 for w in DEMO_KEYWORDS if w in text),
        "course": sum(1 for w in COURSE_KEYWORDS if w in text),
        "placement": sum(1 for w in PLACEMENT_KEYWORDS if w in text),
        "complaint": sum(1 for w in COMPLAINT_KEYWORDS if w in text),
        "interest": sum(1 for w in INTEREST_KEYWORDS if w in text),
        "availability": sum(1 for w in AVAILABILITY_KEYWORDS if w in text),
        "urgent": sum(1 for w in URGENT_KEYWORDS if w in text),
    }
    # Normalize
    total = sum(intents.values()) or 1
    scores = {k: v / total for k, v in intents.items()}
    primary = max(scores, key=scores.get)
    return {"primary": primary, "scores": scores, "all_intents": intents}


# ── Reply Templates ──────────────────────────────────────────────────────
# Each intent has multiple template variations for natural variety.

TEMPLATES = {
    "pricing": [
        {
            "subject_variants": [
                "Re: {subject} — Pricing Details",
                "Re: {subject}",
                "Following up on your pricing query",
            ],
            "body": """Hi {customer_name},

Thank you for your interest in our programs!

I'd be happy to share our pricing details with you. We have flexible options designed to fit different budgets, including EMI plans starting at just Rs. {emi_start}/month.

Here's a quick overview:
• Short-term courses: Rs. {short_price} - Rs. {short_price_high}
• Full programs: Rs. {full_price} - Rs. {full_price_high}
• We also offer early-bird discounts and scholarship options

I understand choosing the right program is an important decision. Could we schedule a quick 10-minute call to walk you through the options that best fit your goals?

Please reply to this email or call us at {team_phone} — we're happy to help!

Best regards,
{sender_name}""",
        },
        {
            "subject_variants": [
                "Re: {subject} — Let's discuss pricing",
                "Re: {subject}",
            ],
            "body": """Hi {customer_name},

Great to hear from you! I understand you're looking for pricing information.

We keep our pricing transparent and competitive. I've attached our latest fee brochure for your reference.

The best part — we offer        0% EMI options and merit-based scholarships, so cost should not be a barrier to your goals.

Would you like me to walk you through the options? Just reply with a convenient time and I'll set it up.

Looking forward to hearing from you!

Warm regards,
{sender_name}""",
        },
    ],
    "demo": [
        {
            "subject_variants": [
                "Re: {subject} — Schedule your demo",
                "Re: {subject}",
                "Demo request — let's set it up!",
            ],
            "body": """Hi {customer_name},

Thanks for showing interest in experiencing our program firsthand!

I'd love to arrange a personalized demo session for you. This will give you a real feel for:
• Our teaching methodology
• Course content and structure
• Hands-on projects and tools we use

We have demo slots available this week. Would any of these work for you?
• Tomorrow between 10 AM — 12 PM
• Day after tomorrow, 3 PM — 5 PM
• Any weekday this week that suits you

Just reply with your preferred time and I'll confirm the booking immediately.

Talk soon!
{sender_name}""",
        },
    ],
    "course": [
        {
            "subject_variants": [
                "Re: {subject} — Course Details",
                "Re: {subject}",
            ],
            "body": """Hi {customer_name},

Thank you for reaching out about our courses! I'm glad you're considering us for your learning journey.

Here's a quick snapshot of what we offer:
• Industry-aligned curriculum designed by practitioners
• Flexible batch timings (weekday & weekend options)
• Hands-on projects + real-world case studies
• Dedicated placement assistance

The next batch starts soon, and seats are filling up fast. I'd love to share the complete syllabus and help you pick the right program for your career goals.

Could we connect for a quick chat today or tomorrow? Reply with a time that works for you, or call us at {team_phone}.

Happy to help anytime!

Best,
{sender_name}""",
        },
    ],
    "placement": [
        {
            "subject_variants": [
                "Re: {subject} — Career & Placement Info",
                "Re: {subject}",
            ],
            "body": """Hi {customer_name},

That's a great question — and one we take very seriously!

Our placement record speaks for itself:
• 90%+ placement rate within 3 months of completion
• Partnerships with 200+ companies
• Dedicated career support: resume building, mock interviews, and direct referrals

I'd love to share specific details about placements in your area of interest. Could we schedule a quick call to discuss this?

Feel free to reply here or reach us at {team_phone} — we're just a call away.

Looking forward to helping you land your dream role!

Best regards,
{sender_name}""",
        },
    ],
    "complaint": [
        {
            "subject_variants": [
                "Re: {subject} — We're here to help",
                "Re: {subject}",
                "Following up on your concern",
            ],
            "body": """Hi {customer_name},

Thank you for bringing this to our attention. I sincerely apologize for the inconvenience you've experienced — this is not the standard we hold ourselves to.

Your concern is important to us, and I want to make sure this gets resolved right away. I've already escalated this to our team lead, and we'll get back to you within the next 2 hours with a solution.

In the meantime, if you'd like to discuss this directly, please call us at {team_phone} — ask for the team lead and they'll assist you immediately.

We value your trust in us and will make this right.

Sincerely,
{sender_name}""",
        },
    ],
    "interest": [
        {
            "subject_variants": [
                "Re: {subject} — Thanks for your interest!",
                "Re: {subject}",
            ],
            "body": """Hi {customer_name},

Thank you for reaching out! It's great to hear from you.

I'd love to learn more about what you're looking for so I can point you in the right direction. Every learner's journey is different, and I want to make sure we find the perfect fit for you.

Could you tell me:
1. What's your current background?
2. What are your career goals?
3. Any specific area you're most interested in?

Once I know a bit more, I can share the most relevant options and next steps.

Feel free to reply here or give us a quick call at {team_phone} — we're happy to chat anytime!

Warm regards,
{sender_name}""",
        },
    ],
    "availability": [
        {
            "subject_variants": [
                "Re: {subject} — Batch Schedule & Availability",
                "Re: {subject}",
            ],
            "body": """Hi {customer_name},

Great question! Here's the current batch schedule:

• Weekday batches: Mon—Fri, 7:00 PM — 9:30 PM
• Weekend batches: Sat—Sun, 10:00 AM — 1:00 PM
• Fast-track option: Available on request

The next batch starts on {next_start_date}, and we currently have {seats_left} seats remaining.

I'd recommend reserving your spot early since batches fill up quickly. Would you like me to hold a seat for you?

Just reply with your preferred batch and I'll send you the enrollment details right away.

See you in class!
{sender_name}""",
        },
    ],
    "urgent": [
        {
            "subject_variants": [
                "Re: {subject} — On it right away!",
                "Re: {subject}",
            ],
            "body": """Hi {customer_name},

I've received your message and I'm on it right away!

I understand this is urgent and I want to make sure we address it immediately. Our team is available right now and I'll get back to you within the next 30 minutes with a complete response.

If you need immediate assistance, please call us at {team_phone} — someone is available to help you right now.

We take your time seriously and won't keep you waiting.

On it!
{sender_name}""",
        },
    ],
}

# Fallback (when no strong intent detected)
FALLBACK_TEMPLATE = {
    "subject_variants": [
        "Re: {subject}",
        "Following up on your message",
    ],
    "body": """Hi {customer_name},

Thank you for reaching out to us! I really appreciate you taking the time to write.

I've noted your message and want to make sure I give you the most helpful response. One of our team members will get back to you shortly with detailed information.

In the meantime, if you'd like a quicker response, feel free to:
• Reply to this email with any specific questions
• Call us at {team_phone}
• Visit our website for more information

We're here to help and look forward to connecting with you!

Best regards,
{sender_name}""",
}


# ── Reply Generator ──────────────────────────────────────────────────────

def generate_reply(customer_name: str, customer_email: str,
                   subject: str, message_text: str,
                   channel: str = "Email") -> dict:
    """
    Generate a human-like reply based on detected intent.
    Returns dict with subject, body, intent, and metadata.
    """
    intent = detect_intent(message_text)
    primary = intent["primary"]

    # Select template based on intent
    templates = TEMPLATES.get(primary, [FALLBACK_TEMPLATE])
    template = random.choice(templates)

    # Pick subject variant
    subject_template = random.choice(template["subject_variants"])
    reply_subject = subject_template.format(subject=subject)

    # Personalize body
    greeting = random.choice(GREETINGS)
    time_greet = random.choice(TIME_GREETINGS.get(_time_of_day(), ["Hi"]))

    # Vary the opening slightly
    if random.random() < 0.3:
        greeting = time_greet

    # Clean up customer name
    if not customer_name or customer_name == "there":
        customer_name_val = "there"
    else:
        customer_name_val = customer_name.split()[0] if customer_name else "there"

    body = template["body"].format(
        customer_name=customer_name_val,
        sender_name=SENDER_NAME,
        team_phone=TEAM_PHONE,
        subject=subject,
        company=COMPANY_NAME,
        # Pricing placeholders
        emi_start=random.choice(["2,999", "3,499", "4,999"]),
        short_price=random.choice(["15,000", "18,000", "22,000"]),
        short_price_high=random.choice(["25,000", "30,000", "35,000"]),
        full_price=random.choice(["45,000", "55,000", "65,000"]),
        full_price_high=random.choice(["85,000", "95,000", "1,20,000"]),
        # Availability placeholders
        next_start_date="Monday, July 7th",
        seats_left=random.choice(["8", "12", "15", "20"]),
    )

    # Remove auto-generated footer mentions
    body = body.replace(
        "Auto-generated by the Missed-Lead Detector system.",
        ""
    )

    return {
        "reply_subject": reply_subject,
        "reply_body": body,
        "detected_intent": primary,
        "intent_scores": intent["scores"],
        "channel": channel,
        "is_auto_replied": True,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def format_reply_preview(customer_name: str, customer_email: str,
                         subject: str, message_text: str) -> str:
    """Generate a human-readable preview of the reply."""
    reply = generate_reply(customer_name, customer_email, subject, message_text)
    return f"""
{'='*60}
  AUTO-REPLY PREVIEW
{'='*60}
  To      : {customer_name} <{customer_email}>
  Subject : {reply['reply_subject']}
  Intent  : {reply['detected_intent']}
  Auto    : Yes (will appear as manual reply)
{'='*60}
{reply['reply_body']}
{'='*60}
"""


if __name__ == "__main__":
    # Demo: generate replies for various email types
    test_emails = [
        ("Priya", "priya@gmail.com", "Course pricing?",
         "Hi, I want to know the price of your data science course. Do you have EMI options?"),
        ("Rahul", "rahul@outlook.com", "Want a demo",
         "Hello, I'm interested in your program. Can you show me a demo?"),
        ("Ananya", "ananya@gmail.com", "Job placement?",
         "Hi, I'm looking for a course with guaranteed placement. What are your placement records?"),
        ("Vikram", "vikram@yahoo.com", "URGENT query",
         "Hello, I need immediate help. My enrollment is not showing up. This is urgent!"),
        ("Meera", "meera@gmail.com", "General enquiry",
         "Hey, I came across your institute. Can you tell me more about what you offer?"),
    ]

    for name, email, subj, msg in test_emails:
        print(format_reply_preview(name, email, subj, msg))
