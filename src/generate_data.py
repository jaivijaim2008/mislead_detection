"""
generate_data.py — Missed-Lead Detector (Realistic v2)
Generates 500 synthetic CRM/chat leads with realistic overlapping distributions,
noise, and edge cases to simulate genuine business data.
"""

import pandas as pd
import numpy as np
import os

SEED = 42
np.random.seed(SEED)
N = 500
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "leads.csv")

CHANNELS   = ["WhatsApp", "Email", "Website Chat", "Phone Inquiry"]
CHANNEL_PROBS = [0.40, 0.30, 0.20, 0.10]

INTENT_KW  = ["price", "buy", "interested", "demo", "quote", "available", "how much"]
FILLER_KW  = ["hello", "hi", "okay", "thanks", "just checking"]
MIXED_MSG  = ["price available", "interested in pricing", "demo quote please",
              "looking to buy", "checking availability", "how much for demo",
              "interested, tell me more", "want to see demo", "need a quote please"]

HOURS = list(range(8, 21))  # 8 AM – 8 PM


def pick_message(high_intent: bool) -> str:
    """Generate a realistic message with some noise (mixed signals)."""
    # 15% chance of mixed signals (high-intent keyword but casual tone, or vice versa)
    if np.random.rand() < 0.15:
        return np.random.choice(MIXED_MSG)

    if high_intent:
        pool = INTENT_KW
    else:
        pool = FILLER_KW

    n_words = np.random.randint(1, 4)
    words = np.random.choice(pool, size=n_words, replace=True)

    # 10% chance high-intent messages also include a filler word (noise)
    if high_intent and np.random.rand() < 0.10:
        filler = np.random.choice(FILLER_KW, size=1)[0]
        words = np.append(words, filler)
        np.random.shuffle(words)

    return " ".join(words)


def _rand_gap_missed() -> float:
    """Realistic: most missed leads have gaps 8-100h, some much longer."""
    gap = np.random.lognormal(mean=3.2, sigma=0.9) * 2.5  # median ~30h, tail up to 300h
    return min(max(gap, 3.0), 360.0)


def _rand_gap_replied() -> float:
    """Realistic: most replied leads have gaps 0.2-24h, some longer if busy."""
    # 85% fast replies (< 18h), 15% slow (staff was busy)
    if np.random.rand() < 0.85:
        gap = np.random.lognormal(mean=1.8, sigma=0.9)  # median ~6h
    else:
        gap = np.random.uniform(18, 72)  # delayed replies
    return round(max(gap, 0.1), 2)


def generate() -> pd.DataFrame:
    rows = []
    base_missed_rate = 0.22  # ~22% baseline missed

    for i in range(N):
        channel = np.random.choice(CHANNELS, p=CHANNEL_PROBS)
        msg_hour = np.random.choice(HOURS)

        # --- Determine high intent ---
        # Intent probability varies by channel (WhatsApp & Phone tend to be higher intent)
        if channel in ("WhatsApp", "Phone Inquiry"):
            high_intent = np.random.rand() < 0.65
        else:
            high_intent = np.random.rand() < 0.50

        message_text = pick_message(high_intent)
        msg_len = np.random.randint(10, 300)

        # --- Previous contacts ---
        # Loyal customers more likely to be replied, but also more frequently contact
        prev_contacts = np.random.poisson(lam=1.5)
        prev_contacts = min(prev_contacts, 8)

        # --- Is it after hours? (business hours 9-18) ---
        after_hours = int(not (9 <= msg_hour <= 18))

        # --- Probability of being missed ---
        # Factors: intent, after-hours, prev_contacts (loyalty), channel volume, noise
        missed_prob = base_missed_rate

        # High-intent leads are LESS likely to be missed (staff prioritizes them),
        # but still can be missed if overwhelmed — reduces by 30%
        if high_intent:
            missed_prob *= 0.7
        else:
            missed_prob *= 1.3  # Low intent more likely ignored

        # After-hours: 40% more likely to be missed
        if after_hours:
            missed_prob *= 1.4

        # Loyal customers (3+ prev contacts): less likely missed
        if prev_contacts >= 3:
            missed_prob *= 0.8

        # Busy channels (WhatsApp, Email) have slightly higher miss rates
        if channel in ("WhatsApp", "Email"):
            missed_prob *= 1.15

        # Clip and add some random noise
        missed_prob = min(max(missed_prob, 0.05), 0.70)
        missed_roll = np.random.rand()

        # Edge case: 3% chance of random flip (pure noise)
        if np.random.rand() < 0.03:
            replied = 0 if np.random.rand() < 0.5 else 1
        else:
            replied = 0 if missed_roll < missed_prob else 1

        # --- Response gap with overlapping distributions ---
        if replied == 0:  # Missed
            gap = _rand_gap_missed()
        else:
            gap = _rand_gap_replied()

        # Small chance (2%) of extreme outliers in either direction
        if np.random.rand() < 0.02:
            gap = round(np.random.uniform(0.1, 300), 2)
            replied = 0 if gap > 48 else 1

        rows.append({
            "lead_id"          : f"L{i+1:04d}",
            "channel"          : channel,
            "message_text"     : message_text,
            "message_hour"     : msg_hour,
            "message_length"   : msg_len,
            "high_intent_flag" : int(high_intent),
            "prev_contacts"    : prev_contacts,
            "response_gap_hrs" : round(gap, 2),
            "replied"          : replied,
        })

    df = pd.DataFrame(rows)

    # Summary statistics
    missed_count = (df['replied'] == 0).sum()
    replied_count = (df['replied'] == 1).sum()
    print(f"[generate_data] Saved {len(df)} rows -> {OUT}")
    print(f"  Missed leads : {missed_count} ({missed_count/len(df)*100:.1f}%)")
    print(f"  Replied      : {replied_count} ({replied_count/len(df)*100:.1f}%)")

    # Show gap distribution overlap
    missed_gaps = df[df['replied'] == 0]['response_gap_hrs']
    replied_gaps = df[df['replied'] == 1]['response_gap_hrs']
    print(f"\n  Response Gap Stats:")
    print(f"    Missed  — mean={missed_gaps.mean():.1f}h, median={missed_gaps.median():.1f}h, "
          f"min={missed_gaps.min():.1f}h, max={missed_gaps.max():.1f}h")
    print(f"    Replied — mean={replied_gaps.mean():.1f}h, median={replied_gaps.median():.1f}h, "
          f"min={replied_gaps.min():.1f}h, max={replied_gaps.max():.1f}h")

    # Show channel breakdown
    print(f"\n  By Channel:")
    for ch in CHANNELS:
        ch_df = df[df['channel'] == ch]
        ch_missed = (ch_df['replied'] == 0).sum()
        print(f"    {ch:15s}: {len(ch_df):3d} leads, {ch_missed:3d} missed ({ch_missed/len(ch_df)*100:.1f}%)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    generate()
