"""
generate_data.py — Missed-Lead Detector
Generates realistic synthetic lead data for model training.

Features:
  - 4 channels: Email, Phone Inquiry, Website Chat, WhatsApp
  - Intent keywords embedded in message text
  - Realistic response gaps (0-200 hours)
  - Binary target: replied (1) or missed (0)

Output: data/leads.csv
"""

import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Templates ──────────────────────────────────────────────

CHANNELS = ["Email", "Phone Inquiry", "Website Chat", "WhatsApp"]

# Message templates by intent type
MESSAGES = {
    "pricing": [
        "Hi, I want to know the price of your data science course. Do you have EMI options?",
        "Hello, what is the fee structure for the full stack development program?",
        "Can you share pricing details? I'm looking for something affordable.",
        "How much does the digital marketing course cost? Any discounts available?",
        "I'm interested in your courses. What are the charges?",
        "What is the total fee? Do you offer installment plans?",
        "Looking for pricing info on your MBA prep course. Budget is around 40k.",
        "Could you send me a price quote for the HR certification program?",
    ],
    "demo": [
        "Can I get a demo of your teaching before I enroll?",
        "I'd like to try a sample class. Is that possible?",
        "Show me how your online classes work. I want to see a preview.",
        "Do you offer free trial sessions? I want to test the quality.",
        "Can you walk me through the course content first?",
        "I want to attend a demo class before making a decision.",
    ],
    "course": [
        "Tell me about your data science curriculum. What topics do you cover?",
        "What is included in the full stack development program?",
        "Do you have weekend batches? I'm working professional.",
        "How long is the course duration? What is the schedule?",
        "What certifications do you provide after completion?",
        "Can you share the syllabus for the digital marketing course?",
        "I want to know about your course structure and modules.",
    ],
    "placement": [
        "Do you provide placement assistance after the course?",
        "What is your placement rate? Which companies hire from you?",
        "I'm looking for a course with guaranteed job placement.",
        "Do you help with resume building and interview preparation?",
        "Can you share placement statistics for the data science program?",
        "I need a course that offers internship opportunities.",
    ],
    "complaint": [
        "I'm unhappy with the service. Nobody responded to my query.",
        "This is terrible. I've been waiting for 3 days for a response.",
        "Very disappointed with the support. Not recommending to anyone.",
        "The quality is poor. I want a refund.",
        "I've had a bad experience. Please resolve this immediately.",
    ],
    "interest": [
        "I'm interested in learning data science. Can you help?",
        "Looking for a good course to upskill. What do you recommend?",
        "I want to switch careers into tech. Where should I start?",
        "Can you tell me more about your programs?",
        "I came across your institute. Want to know more.",
        "Hey, I need some information about your courses.",
    ],
    "availability": [
        "When does the next batch start? I want to join ASAP.",
        "What are the class timings? Are weekend slots available?",
        "Do you have any seats left for the upcoming batch?",
        "I want to start next week. Is that possible?",
        "How long is the course? When can I enroll?",
        "What is the duration of the program?",
    ],
    "urgent": [
        "URGENT: I need immediate response. My enrollment is pending!",
        "This is urgent! I need help right now. Please respond ASAP!",
        "I need this resolved today. It's been 2 days already!",
        "EMERGENCY: My payment is stuck. Need immediate assistance!",
        "Please respond immediately. I'm losing patience!",
    ],
}

# Fallback messages (low intent)
FALLBACK_MESSAGES = [
    "Hello, just checking what you offer.",
    "Hi, I came across your website.",
    "Can you tell me something about your institute?",
    "Hey there, just browsing.",
    "I saw your ad. What is this about?",
    "Just looking around. No specific question.",
    "Hi, what services do you provide?",
    "Hello, I'm just exploring options.",
]

CUSTOMER_NAMES = [
    "Priya Sharma", "Rahul Patel", "Ananya Singh", "Vikram Kumar",
    "Meera Reddy", "Arjun Nair", "Sneha Gupta", "Karthik Menon",
    "Deepa Iyer", "Rohit Verma", "Nisha Agarwal", "Sanjay Das",
    "Pooja Joshi", "Amit Choudhary", "Divya Rao", "Suresh Babu",
    "Lakshmi Prasad", "Rajesh Tiwari", "Kavitha Sundaram", "Manoj Kumar",
    "Swathi Venkatesh", "Vivek Sharma", "Anitha Raj", "Prakash Reddy",
    "Shruti Mishra", "Ganesh Pillai", "Revathi Krishnan", "Sunil Mehta",
    "Geeta Devi", "Ramesh Chandra",
]


def generate_leads(n: int = 5000) -> pd.DataFrame:
    """Generate n synthetic leads with realistic distributions."""
    rows = []

    for i in range(1, n + 1):
        lead_id = f"L{i:05d}"
        channel = random.choice(CHANNELS)
        name = random.choice(CUSTOMER_NAMES)

        # Decide intent type (weighted: more interest/pricing, fewer complaints)
        intent_weights = {
            "pricing": 0.18,
            "interest": 0.20,
            "course": 0.15,
            "availability": 0.12,
            "demo": 0.10,
            "placement": 0.10,
            "complaint": 0.05,
            "urgent": 0.03,
            "fallback": 0.07,
        }
        intent_type = random.choices(
            list(intent_weights.keys()),
            weights=list(intent_weights.values()),
            k=1,
        )[0]

        if intent_type == "fallback":
            message_text = random.choice(FALLBACK_MESSAGES)
            high_intent = 0
        else:
            message_text = random.choice(MESSAGES[intent_type])
            high_intent = 1

        # Message hour: weighted towards business hours
        hour_weights = [0.03, 0.05, 0.06, 0.08, 0.10, 0.10, 0.08, 0.08,
                        0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02,
                        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        message_hour = random.choices(range(24), weights=hour_weights, k=1)[0]

        message_length = len(message_text)

        # Previous contacts: most are 0-2
        prev_contacts = random.choices([0, 1, 2, 3, 4, 5],
                                        weights=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05],
                                        k=1)[0]

        # Response gap: depends on intent and whether they'll be replied to
        # Higher intent = smaller expected gap for replied leads
        if high_intent:
            base_gap = random.expovariate(1 / 30)  # mean ~30h
        else:
            base_gap = random.expovariate(1 / 60)  # mean ~60h
        response_gap_hrs = round(min(base_gap + random.gauss(0, 5), 300), 2)
        response_gap_hrs = max(0.1, response_gap_hrs)

        # Target: replied (1) or missed (0)
        # Factors: higher gap → more likely missed, high intent → more likely replied
        miss_prob = 0.3  # base miss rate
        if response_gap_hrs > 48:
            miss_prob += 0.3
        elif response_gap_hrs > 24:
            miss_prob += 0.15
        elif response_gap_hrs < 6:
            miss_prob -= 0.2

        if high_intent:
            miss_prob -= 0.15

        if channel in ["Phone Inquiry", "WhatsApp"]:
            miss_prob -= 0.05  # easier to respond on these

        miss_prob = max(0.05, min(0.95, miss_prob))
        replied = 0 if random.random() < miss_prob else 1

        rows.append({
            "lead_id": lead_id,
            "channel": channel,
            "message_text": message_text,
            "message_hour": message_hour,
            "message_length": message_length,
            "high_intent_flag": high_intent,
            "prev_contacts": prev_contacts,
            "response_gap_hrs": response_gap_hrs,
            "replied": replied,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  MISSED-LEAD DETECTOR - SYNTHETIC DATA GENERATOR")
    print("=" * 60)

    df = generate_leads(n=5000)
    out_path = os.path.join(DATA_DIR, "leads.csv")
    df.to_csv(out_path, index=False)

    print(f"\n[generate] Generated {len(df)} leads -> {out_path}")
    print(f"[generate] Channel distribution:")
    print(df["channel"].value_counts().to_string())
    print(f"\n[generate] Target distribution:")
    print(f"  Replied:  {df['replied'].sum()} ({df['replied'].mean():.1%})")
    print(f"  Missed:   {(df['replied'] == 0).sum()} ({(df['replied'] == 0).mean():.1%})")
    print(f"\n[generate] High intent: {df['high_intent_flag'].sum()} ({df['high_intent_flag'].mean():.1%})")
    print(f"[generate] Avg response gap: {df['response_gap_hrs'].mean():.1f}h")
    print(f"\n[generate] Done!")
