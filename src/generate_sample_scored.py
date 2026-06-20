"""
generate_sample_scored.py — Creates a realistic sample leads_scored.csv
with Gmail-specific columns so the dashboard works on Streamlit Cloud.

Run: python src/generate_sample_scored.py
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)
N = 40

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "outputs", "leads_scored.csv")

# Realistic customer data
customers = [
    ("Priya Sharma", "priya.sharma@gmail.com"),
    ("Rahul Patel", "rahul.patel@outlook.com"),
    ("Ananya Krishnan", "ananya.k@rediffmail.com"),
    ("Vikram Singh", "vikram.singh@yahoo.co.in"),
    ("Meera Nair", "meera.nair@gmail.com"),
    ("Arjun Reddy", "arjun.reddy@company.com"),
    ("Deepa Iyer", "deepa.iyer@gmail.com"),
    ("Karthik Menon", "karthik.m@outlook.com"),
    ("Sneha Gupta", "sneha.gupta@gmail.com"),
    ("Ravi Kumar", "ravi.kumar@company.com"),
    ("Lakshmi Venkatesh", "lakshmi.v@gmail.com"),
    ("Suresh Babu", "suresh.b@rediffmail.com"),
    ("Nandhini Raj", "nandhini.raj@gmail.com"),
    ("Prakash Verma", "prakash.v@outlook.com"),
    ("Divya Subramanian", "divya.s@gmail.com"),
    ("Mohan Das", "mohan.das@yahoo.co.in"),
    ("Kavitha Raman", "kavitha.r@gmail.com"),
    ("Arun Thiru", "arun.thiru@company.com"),
    ("Revathi Sundaram", "revathi.s@gmail.com"),
    ("Ganesh Pillai", "ganesh.p@outlook.com"),
]

subjects = [
    "Interested in your course offerings",
    "Request for pricing information",
    "Need details about MBA program",
    "Want to know about placement assistance",
    "Enquiry about batch schedule",
    "Looking for part-time options",
    "Query about EMI payment plans",
    "Need brochure for Digital Marketing course",
    "Interested in Data Science certification",
    "Asking about weekend batch availability",
    "Question about internship opportunities",
    "Want to compare with other institutes",
    "Enquiring about scholarship options",
    "Need information on PGP course",
    "Looking for HR certification course",
]

messages = [
    "Hi, I'm interested in your courses. Could you please share the fee structure and batch timings?",
    "Hello, I saw your ad on LinkedIn. I want to know more about the MBA program. Please share details.",
    "Good morning, I'm looking for a career switch. Can you tell me about your Data Science course?",
    "I'm a working professional. Do you have weekend batches for the Digital Marketing course?",
    "Hi, my friend recommended your institute. I'd like to know about placement records.",
    "Hello, I'm interested in the PGP program. What is the total fees and duration?",
    "Can you share the syllabus for the Business Analytics course? Also need info on EMI options.",
    "I want to enroll in the next batch. When does it start? Please share details.",
    "Hi, I need to know if you offer online classes. I'm based in Bangalore.",
    "Interested in your HR certification. What are the job prospects after completion?",
    "Please share pricing for the Full Stack Development course. Looking for EMI options.",
    "Hello, I'm a recent graduate. Which course would you recommend for placement?",
    "Hi, I want to know about your campus facility. Can I visit this weekend?",
    "I'm comparing institutes. What makes your program different? Please share details.",
    "Looking for a part-time MBA. Do you offer flexible scheduling?",
]

# Generate data
rows = []
now = datetime(2026, 6, 15, 10, 0, 0)
lead_counter = 0

for i in range(N):
    customer_name, customer_email = customers[i % len(customers)]
    subject = subjects[i % len(subjects)]
    message = messages[i % len(messages)]

    # Randomize timing: 1–60 days ago
    days_ago = np.random.randint(1, 60)
    hours_ago = np.random.randint(0, 23)
    received = now - timedelta(days=days_ago, hours=hours_ago)

    # Response gap: missed leads have higher gaps
    is_missed_candidate = np.random.random() < 0.45
    if is_missed_candidate:
        response_gap = np.random.uniform(24, 360)
    else:
        response_gap = np.random.uniform(0.5, 18)

    # High intent: from message content
    intent_keywords = ["interested", "pricing", "fee", "enroll", "buy", "want", "need", "looking"]
    msg_lower = message.lower()
    high_intent = int(any(kw in msg_lower for kw in intent_keywords))

    # ML prediction — skew toward ~30% missed for a good demo
    missed_prob = min(0.97, max(0.03,
        0.4 * (response_gap / 200) +
        0.25 * (1 - high_intent) +
        0.2 * np.random.random() +
        0.15 * (days_ago > 20)
    ))
    predicted_missed = int(missed_prob >= 0.45)

    lead_counter += 1
    lead_id = f"GMB_{lead_counter:04d}"

    rows.append({
        "lead_id": lead_id,
        "_customer_name": customer_name,
        "_customer_email": customer_email,
        "_subject": subject,
        "message_text": message,
        "message_hour": received.hour,
        "message_length": len(message),
        "high_intent_flag": high_intent,
        "prev_contacts": np.random.randint(0, 6),
        "response_gap_hrs": round(response_gap, 2),
        "missed_probability": round(missed_prob, 4),
        "predicted_missed": predicted_missed,
        "_received_time": received.strftime("%Y-%m-%d %H:%M:%S"),
    })

df = pd.DataFrame(rows)

# Save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Generated {len(df)} sample Gmail-scored leads -> {OUT}")
print(f"  Columns: {list(df.columns)}")
print(f"  Missed: {df['predicted_missed'].sum()}/{len(df)}")
print(f"  High Intent: {df['high_intent_flag'].sum()}/{len(df)}")
print(f"  Avg Gap: {df['response_gap_hrs'].mean():.1f}h")
