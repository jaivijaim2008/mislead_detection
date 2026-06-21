"""
merge_kaggle_data.py — Missed-Lead Detector
Merges real-world Kaggle datasets into the unified training schema.

Datasets:
  1. X Education Lead Scoring (9,240 CRM leads)
  2. Customer Support Intent (4,000 support tickets)

Unified Schema:
  lead_id, channel, message_text, message_hour, message_length,
  high_intent_flag, prev_contacts, response_gap_hrs, replied

"""

import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(__file__)
KAGGLE_DIR = os.path.join(BASE, "..", "data", "kaggle")
DATA_DIR = os.path.join(BASE, "..", "data")


# ─────────────────────────────────────────────────────────
# 1. LOAD RAW DATASETS
# ─────────────────────────────────────────────────────────
def load_lead_scoring():
    """Load X Education Lead Scoring dataset (9,240 real CRM leads)."""
    path = os.path.join(KAGGLE_DIR, "Lead Scoring.csv")
    df = pd.read_csv(path)
    print(f"[merge] Loaded Lead Scoring: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_support_tickets():
    """Load Customer Support Intent dataset (4,000 real tickets)."""
    # Use the 3.4k manageable version
    path = os.path.join(KAGGLE_DIR, "dataset-tickets-multi-lang3-4k.csv")
    df = pd.read_csv(path)
    print(f"[merge] Loaded Support Tickets: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────
# 2. MAP LEAD SCORING → UNIFIED SCHEMA
# ─────────────────────────────────────────────────────────
def map_lead_scoring(df):
    """
    Map X Education Lead Scoring features to unified schema.

    Feature mapping:
      Lead Source       → channel (mapped to our categories)
      TotalVisits       → prev_contacts
      Total Time Spent → response_gap_hrs (inverted: more time = less gap)
      Last Activity     → high_intent_flag
      Converted         → replied (target)
      Prospect ID       → lead_id
      We synthesize message_text, message_hour, message_length from available data.
    """
    result = pd.DataFrame()

    # Lead ID
    result["lead_id"] = ["KLS_" + str(i).zfill(5) for i in range(len(df))]

    # Channel mapping: Lead Source → our channel categories
    source_channel_map = {
        "google": "Email",
        "bing": "Email",
        "google ads": "Email",
        "facebook": "WhatsApp",
        "youtube": "Website Chat",
        "linkedin": "Email",
        "twitter": "WhatsApp",
        "direct traffic": "Phone Inquiry",
        "organic search": "Email",
        "reference": "Phone Inquiry",
        "welingak website": "Website Chat",
        "referral sites": "Website Chat",
        "olark chat": "Website Chat",
        "live chat": "Website Chat",
    }
    result["channel"] = df["Lead Source"].fillna("unknown").str.lower().map(
        source_channel_map
    ).fillna("Email")

    # Message text synthesis from available text fields
    def synthesize_message(row):
        parts = []
        if pd.notna(row.get("What is your current occupation")):
            parts.append(row["What is your current occupation"])
        if pd.notna(row.get("What matters most to you in choosing a course")):
            parts.append(row["What matters most to you in choosing a course"])
        if pd.notna(row.get("Tags")):
            parts.append(row["Tags"])
        if not parts:
            # Synthesize from numeric features
            if row.get("TotalVisits", 0) > 5:
                parts.append("interested in demo how much")
            elif row.get("Total Time Spent on Website", 0) > 500:
                parts.append("price quote available")
            else:
                parts.append("just checking")
        return " ".join(parts[:3]).lower()

    result["message_text"] = df.apply(synthesize_message, axis=1)

    # Message hour: synthesize from Last Activity timestamps or random business hours
    np.random.seed(42)
    result["message_hour"] = np.random.choice(
        range(8, 21), size=len(df),
        p=[0.05, 0.08, 0.10, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.07,
           0.06, 0.05, 0.05]
    )

    # Message length: derived from text synthesis
    result["message_length"] = result["message_text"].str.len()

    # High intent flag: based on Total Time Spent + Total Visits
    time_spent = df["Total Time Spent on Website"].fillna(0)
    total_visits = df["TotalVisits"].fillna(0)
    result["high_intent_flag"] = ((time_spent > 300) | (total_visits > 3)).astype(int)

    # Previous contacts: TotalVisits mapped
    result["prev_contacts"] = df["TotalVisits"].fillna(0).astype(int)

    # Response gap hours: inverse of Total Time Spent (more time = less gap)
    # Also factor in Last Activity recency
    time_spent_norm = time_spent / (time_spent.max() + 1)  # 0 to 1
    result["response_gap_hrs"] = np.clip(
        (1 - time_spent_norm) * 100 + np.random.normal(0, 5, len(df)),
        0.1, 360
    ).round(2)

    # Target: Converted → replied (1=replied, 0=missed)
    # In our schema: replied=1 means they replied (good), replied=0 means missed
    # In Lead Scoring: Converted=1 means converted (good), Converted=0 means didn't
    result["replied"] = df["Converted"].fillna(0).astype(int)

    print(f"[merge] Lead Scoring mapped: {len(result)} rows, "
          f"replied rate: {result['replied'].mean():.1%}")
    return result


# ─────────────────────────────────────────────────────────
# 3. MAP SUPPORT TICKETS → UNIFIED SCHEMA
# ─────────────────────────────────────────────────────────
def map_support_tickets(df):
    """
    Map Customer Support Intent features to unified schema.

    Feature mapping:
      body/subject    → message_text
      type            → channel (Incident→Email, Request→WhatsApp, etc.)
      priority        → high_intent_flag (high=1, low=0)
      tag_1           → prev_contacts (count of tags as proxy)
      queue           → additional channel signal
      answer presence → replied (target)
    """
    result = pd.DataFrame()

    # Lead ID
    result["lead_id"] = ["KCS_" + str(i).zfill(5) for i in range(len(df))]

    # Channel mapping: type → our categories
    type_channel_map = {
        "Incident": "Email",
        "Request": "WhatsApp",
        "Problem": "Phone Inquiry",
        "Change Request": "Website Chat",
        "Question": "Website Chat",
    }
    result["channel"] = df["type"].fillna("Incident").map(type_channel_map).fillna("Email")

    # Message text: combine subject + body
    def combine_text(row):
        parts = []
        if pd.notna(row.get("subject")):
            parts.append(str(row["subject"]))
        if pd.notna(row.get("body")):
            body = str(row["body"])[:200]  # Truncate long bodies
            parts.append(body)
        return " ".join(parts).lower() if parts else "customer inquiry"

    result["message_text"] = df.apply(combine_text, axis=1)

    # Message hour: synthesize from priority patterns
    np.random.seed(123)
    result["message_hour"] = np.random.choice(
        range(8, 21), size=len(df),
        p=[0.05, 0.08, 0.10, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.07,
           0.06, 0.05, 0.05]
    )

    # Message length
    result["message_length"] = result["message_text"].str.len()

    # High intent flag: based on priority
    priority_map = {"high": 1, "medium": 1, "low": 0}
    result["high_intent_flag"] = df["priority"].fillna("medium").str.lower().map(
        priority_map
    ).fillna(0).astype(int)

    # Previous contacts: count non-null tags as proxy
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    result["prev_contacts"] = df[tag_cols].notna().sum(axis=1)

    # Response gap hours: based on priority (high priority = smaller gap expected)
    base_gap = {"high": 8, "medium": 24, "low": 72}
    result["response_gap_hrs"] = df["priority"].fillna("medium").str.lower().map(
        base_gap
    ).fillna(24) + np.random.normal(0, 5, len(df))
    result["response_gap_hrs"] = result["response_gap_hrs"].clip(0.1, 360).round(2)

    # Target: if answer exists → replied=1, else replied=0
    result["replied"] = df["answer"].notna().astype(int)

    print(f"[merge] Support Tickets mapped: {len(result)} rows, "
          f"replied rate: {result['replied'].mean():.1%}")
    return result


# ─────────────────────────────────────────────────────────
# 4. MERGE WITH EXISTING SYNTHETIC DATA
# ─────────────────────────────────────────────────────────
def merge_all():
    """Merge all three datasets into unified training data."""
    # Load original synthetic data
    synthetic_path = os.path.join(DATA_DIR, "leads.csv")
    if not os.path.exists(synthetic_path):
        print(f"[merge] No synthetic data found at {synthetic_path}")
        print("[merge] Run generate_data.py first, or run this script standalone to merge only Kaggle datasets.")
        df_synth = pd.DataFrame(columns=[
            "lead_id", "channel", "message_text", "message_hour",
            "message_length", "high_intent_flag", "prev_contacts",
            "response_gap_hrs", "replied"
        ])
    else:
        df_synth = pd.read_csv(synthetic_path)
    print(f"[merge] Original synthetic data: {len(df_synth)} rows")

    # Load and map Kaggle datasets
    df_lead_raw = load_lead_scoring()
    df_ticket_raw = load_support_tickets()

    df_lead = map_lead_scoring(df_lead_raw)
    df_ticket = map_support_tickets(df_ticket_raw)

    # Ensure all datasets have the same columns
    unified_cols = [
        "lead_id", "channel", "message_text", "message_hour",
        "message_length", "high_intent_flag", "prev_contacts",
        "response_gap_hrs", "replied"
    ]

    df_synth = df_synth[unified_cols]
    df_lead = df_lead[unified_cols]
    df_ticket = df_ticket[unified_cols]

    # Combine: synthetic + Kaggle Lead Scoring + Kaggle Support Tickets
    df_merged = pd.concat([df_synth, df_lead, df_ticket], ignore_index=True)

    # Remove duplicates if any
    df_merged = df_merged.drop_duplicates(subset=["lead_id"], keep="first")

    # Save merged dataset
    merged_path = os.path.join(DATA_DIR, "leads_merged.csv")
    df_merged.to_csv(merged_path, index=False)

    print(f"\n[merge] === MERGED DATASET ===")
    print(f"  Total rows: {len(df_merged)}")
    print(f"  Synthetic:  {len(df_synth)}")
    print(f"  Lead Scoring: {len(df_lead)}")
    print(f"  Support Tickets: {len(df_ticket)}")
    print(f"  Channels: {df_merged['channel'].value_counts().to_dict()}")
    print(f"  Replied rate: {df_merged['replied'].mean():.1%}")
    print(f"  Saved to: {merged_path}")

    return df_merged


if __name__ == "__main__":
    print("=" * 60)
    print("  KAGGLE DATASET MERGE")
    print("=" * 60)
    merge_all()
    print("\n[merge] Done!")
