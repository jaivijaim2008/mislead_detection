"""
orchestrator.py — Missed-Lead Detector
Daily pipeline runner. Ties all modules together.

Modes:
  python src/orchestrator.py              ← score synthetic CSV data (demo)
  python src/orchestrator.py --live       ← fetch real emails from Gmail inbox & send real follow-ups
  python src/orchestrator.py --preview    ← just preview real inbox emails without sending

"""

import os, sys, pickle, time
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from auto_followup     import process_missed_leads
from employee_reminder import start_reminder, mark_replied

BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "..", "data",   "leads.csv")
MODELS  = os.path.join(BASE, "..", "models")
OUT     = os.path.join(BASE, "..", "outputs","leads_scored.csv")

GAP_THRESHOLD_HRS = 24


def load_artefacts():
    """Load ML ensemble + DL model for grand ensemble scoring."""
    with open(os.path.join(MODELS, "ensemble.pkl"), "rb") as f:
        ensemble = pickle.load(f)
    with open(os.path.join(MODELS, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Try loading DL model (optional — falls back to ML-only if unavailable)
    dl_model = None
    dl_model_path = os.path.join(MODELS, "dl_model.pt")
    dl_scaler_path = os.path.join(MODELS, "dl_scaler.pkl")
    if os.path.exists(dl_model_path) and os.path.exists(dl_scaler_path):
        try:
            import torch
            from deep_learning import load_dl_model, DEVICE
            dl_model, dl_scaler = load_dl_model(dl_model_path, dl_scaler_path)
            print("[pipeline] Deep Learning model loaded on", DEVICE)
        except Exception as e:
            print(f"[pipeline] DL model load failed ({e}), using ML-only scoring")
            dl_model = None
            dl_scaler = None
    else:
        dl_scaler = None
        print("[pipeline] No DL model found, using ML-only scoring")

    return ensemble, scaler, dl_model, dl_scaler


def score_leads(df, ensemble, scaler, dl_model=None, dl_scaler=None):
    """Score leads using grand ensemble (ML + DL) if DL model available."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["channel_enc"]       = le.fit_transform(df["channel"])
    intent_words            = ["price","buy","interested","demo","quote","available"]
    df["intent_score"]      = df["message_text"].apply(
        lambda t: sum(w in str(t).lower() for w in intent_words))
    df["is_business_hours"] = df["message_hour"].between(9, 18).astype(int)
    df["gap_bucket"]        = pd.cut(df["response_gap_hrs"],
                                     bins=[0,6,12,24,9999], labels=[0,1,2,3]).astype(int)
    features = ["channel_enc","message_length","high_intent_flag","prev_contacts",
                "response_gap_hrs","intent_score","is_business_hours",
                "gap_bucket","message_hour"]
    X = scaler.transform(df[features])

    # ML ensemble predictions
    ml_probs = ensemble.predict_proba(X)[:, 1]

    # DL model predictions (if available)
    if dl_model is not None:
        if dl_scaler is not None and dl_scaler is not scaler:
            X_dl = dl_scaler.transform(df[features])
        else:
            X_dl = X
        dl_probs = dl_model.predict_proba_np(X_dl)[:, 1]
        # Grand ensemble: 50/50 soft voting
        df["missed_probability"] = 0.5 * ml_probs + 0.5 * dl_probs
        df["_ml_prob"] = ml_probs
        df["_dl_prob"] = dl_probs
        print(f"[pipeline] Scored with Grand Ensemble (ML+DL)")
    else:
        df["missed_probability"] = ml_probs
        print(f"[pipeline] Scored with ML Ensemble only")

    df["predicted_missed"] = (df["missed_probability"] >= 0.5).astype(int)
    return df


def run_live_pipeline(preview_only: bool = False):
    """
    Fetch real customer emails from Gmail inbox, score them with the model,
    and send real follow-up emails to detected missed leads.
    """
    print("=" * 60)
    print("  MISSED-LEAD DETECTOR - LIVE EMAIL PIPELINE")
    print("=" * 60)

    # Import email reader
    from email_reader import fetch_customer_emails

    df = fetch_customer_emails(max_emails=50, search_since_days=30)
    if df.empty:
        print("[pipeline] No customer emails found. Exiting.")
        return

    ensemble, scaler, dl_model, dl_scaler = load_artefacts()
    df = score_leads(df, ensemble, scaler, dl_model, dl_scaler)

    # Save scored results
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[pipeline] Scored results saved -> {OUT}")

    missed = df[df["predicted_missed"] == 1].copy()
    print(f"\n[pipeline] Model detected {len(missed)} missed leads out of {len(df)} emails")

    # Show scored preview
    if len(missed) > 0:
        print(f"\n{'='*70}")
        print(f"  MISSED LEAD DETAILS:")
        print(f"{'='*70}")
        for _, row in missed.iterrows():
            prob = row['missed_probability']
            print(f"  {row['lead_id']} | {row['_customer_email']:30s} | "
                  f"gap={row['response_gap_hrs']:6.1f}h | "
                  f"intent={row['high_intent_flag']} | "
                  f"prob={prob:.0%}")

    # Build email payloads for missed leads — generate smart replies
    from smart_reply_engine import generate_reply
    email_payloads = []
    for _, row in missed.iterrows():
        reply = generate_reply(
            customer_name=row["_customer_name"],
            customer_email=row["_customer_email"],
            subject=row["_subject"],
            message_text=row["message_text"],
            channel=row.get("channel", "Email"),
        )
        email_payloads.append({
            "lead_id"             : row["lead_id"],
            "customer_email"      : row["_customer_email"],
            "customer_name"       : row["_customer_name"],
            "channel"             : "Email",
            "subject"             : row["_subject"],
            "original_message_id" : row.get("_message_id", ""),
            "reply_subject"       : reply["reply_subject"],
            "reply_body"          : reply["reply_body"],
            "detected_intent"     : reply["detected_intent"],
        })

    if preview_only:
        print(f"\n[PREVIEW MODE] Would send {len(email_payloads)} follow-up emails.")
        print(f"[PREVIEW MODE] Run with --live to actually send them.")
    else:
        print(f"\n[pipeline] Sending {len(email_payloads)} follow-up emails via SMTP...")
        process_missed_leads(email_payloads)

        top_missed = missed.nlargest(5, "response_gap_hrs")
        for _, row in top_missed.iterrows():
            start_reminder({
                "lead_id"      : row["lead_id"],
                "customer_name": row["_customer_name"],
                "channel"      : "Email",
                "gap_hrs"      : row["response_gap_hrs"],
            }, interval=60)

    print("\n" + "=" * 60)
    if preview_only:
        print("  LIVE PREVIEW COMPLETE - no emails sent")
    else:
        print("  LIVE PIPELINE COMPLETE")
    print("=" * 60)


def run_demo_pipeline():
    print("=" * 60)
    print("  MISSED-LEAD DETECTOR - DEMO PIPELINE (CSV Data)")
    print("=" * 60)
    if not os.path.exists(DATA):
        print(f"[pipeline] No sample data found at {DATA}")
        print("[pipeline] Use --live to scan real Gmail inbox instead.")
        return
    df = pd.read_csv(DATA)
    print(f"[pipeline] Loaded {len(df)} leads from {DATA}")
    ensemble, scaler, dl_model, dl_scaler = load_artefacts()
    df = score_leads(df, ensemble, scaler, dl_model, dl_scaler)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    missed = df[df["predicted_missed"] == 1].copy()
    print(f"[pipeline] Detected {len(missed)} missed leads")
    email_payloads = [
        {
            "lead_id"       : row["lead_id"],
            "customer_email": f"{row['lead_id'].lower()}@example.com",
            "customer_name" : "Valued Customer",
            "channel"       : row["channel"],
            "subject"       : "Your Recent Inquiry",
        }
        for _, row in missed.iterrows()
    ]
    process_missed_leads(email_payloads)
    top_missed = missed.nlargest(5, "response_gap_hrs")
    for _, row in top_missed.iterrows():
        start_reminder({
            "lead_id"      : row["lead_id"],
            "customer_name": "Valued Customer",
            "channel"      : row["channel"],
            "gap_hrs"      : row["response_gap_hrs"],
        }, interval=60)
    print("\n[pipeline] Demo run complete.")
    print("=" * 60)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--live" in args:
        run_live_pipeline(preview_only=False)
    elif "--preview" in args:
        run_live_pipeline(preview_only=True)
    else:
        run_demo_pipeline()
