"""
api.py — Missed-Lead Detector
FastAPI REST API exposing scoring, reply preview, lead listing, and scan endpoints.

Run:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    # or
    python src/api.py

Docs:
    http://localhost:8000/docs       (Swagger UI)
    http://localhost:8000/redoc      (ReDoc)
"""
import os
import sys
import time
import pickle
import logging
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure src/ is on path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from api_models import (
    LeadScoreRequest, LeadScoreBatchRequest, LeadScoreResponse,
    ReplyPreviewRequest, ReplyPreviewResponse,
    ReplySendRequest, ReplySendResponse,
    LeadDetailResponse, LeadListResponse,
    StatsResponse, HealthResponse, ErrorResponse,
    ScanRequest, ScanResponse, LeadUpdateRequest,
)

# ── App Setup ──────────────────────────────────────────────

app = FastAPI(
    title="Missed-Lead Detector API",
    description="AI-powered missed lead detection and automated follow-up system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Logging ────────────────────────────────────────────────

LOG_DIR = os.path.join(SRC_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
ACCESS_LOG = os.path.join(LOG_DIR, "api_access.log")

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(ACCESS_LOG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)

# ── Rate Limiter ───────────────────────────────────────────

class RateLimiter:
    """Simple in-memory rate limiter: 100 requests per minute per API key."""
    def __init__(self, max_per_minute: int = 100):
        self.max_per_minute = max_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        self._requests[key] = [t for t in self._requests[key] if now - t < 60]
        if len(self._requests[key]) >= self.max_per_minute:
            return False
        self._requests[key].append(now)
        return True

rate_limiter = RateLimiter(max_per_minute=100)

# ── Auth ───────────────────────────────────────────────────

API_KEY = os.getenv("API_KEY", "mld-dev-key-2024")


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify API key from header."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not rate_limiter.is_allowed(x_api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (100/min)")
    return x_api_key


# ── Model Loading ──────────────────────────────────────────

BASE = os.path.join(SRC_DIR, "..")
MODELS_DIR = os.path.join(BASE, "models")
OUT_DIR = os.path.join(BASE, "outputs")
SCORED_CSV = os.path.join(OUT_DIR, "leads_scored.csv")
REPLY_LOG = os.path.join(LOG_DIR, "auto_replies.json")
FOLLOWUP_LOG = os.path.join(LOG_DIR, "followup_status.json")
SENT_LOG = os.path.join(LOG_DIR, "sent_leads.json")

_ensemble = None
_scaler = None
_models_loaded = False


def load_models():
    """Load ML models (cached)."""
    global _ensemble, _scaler, _models_loaded
    if _models_loaded:
        return
    try:
        ensemble_path = os.path.join(MODELS_DIR, "ensemble.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(ensemble_path) and os.path.exists(scaler_path):
            with open(ensemble_path, "rb") as f:
                _ensemble = pickle.load(f)
            with open(scaler_path, "rb") as f:
                _scaler = pickle.load(f)
            _models_loaded = True
    except Exception as e:
        logger.error(f"Model load failed: {e}")


# ── Helpers ────────────────────────────────────────────────

CHANNEL_MAP = {"email": 0, "phone inquiry": 1, "website chat": 2, "whatsapp": 3}
INTENT_WORDS = ["price", "buy", "interested", "demo", "quote", "available"]


def _load_json(path: str, default=None):
    import json
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default if default is not None else []


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ML features from raw lead data."""
    df["channel_enc"] = df["channel"].apply(
        lambda c: CHANNEL_MAP.get(str(c).lower(), 0))
    df["intent_score"] = df["message_text"].apply(
        lambda t: sum(w in str(t).lower() for w in INTENT_WORDS))
    df["is_business_hours"] = df["message_hour"].between(9, 18).astype(int)
    df["gap_bucket"] = pd.cut(
        df["response_gap_hrs"],
        bins=[0, 6, 12, 24, 9999],
        labels=[0, 1, 2, 3]
    ).astype(int)
    return df


# ── Routes ─────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Missed-Lead Detector API", "docs": "/docs"}


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint (no auth required)."""
    load_models()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=_models_loaded,
        gmail_connected=bool(os.getenv("IMAP_USER")),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/v1/score", response_model=LeadScoreResponse, tags=["Scoring"])
async def score_lead(
    req: LeadScoreRequest,
    api_key: str = Depends(verify_api_key),
):
    """Score a single lead for missed-lead probability."""
    load_models()

    df = pd.DataFrame([{
        "channel": req.channel,
        "message_text": req.message_text,
        "message_hour": req.message_hour,
        "message_length": req.message_length or len(req.message_text),
        "high_intent_flag": req.high_intent_flag if req.high_intent_flag is not None else (
            1 if any(w in req.message_text.lower() for w in INTENT_WORDS) else 0
        ),
        "prev_contacts": req.prev_contacts,
        "response_gap_hrs": req.response_gap_hrs,
    }])

    df = _compute_features(df)

    if _ensemble is not None and _scaler is not None:
        features = ["channel_enc", "message_length", "high_intent_flag", "prev_contacts",
                     "response_gap_hrs", "intent_score", "is_business_hours",
                     "gap_bucket", "message_hour"]
        X = _scaler.transform(df[features])
        prob = float(_ensemble.predict_proba(X)[:, 1][0])
    else:
        # Fallback: heuristic scoring
        prob = min(0.95, 0.3 + 0.2 * (1 - df["high_intent_flag"].iloc[0]) +
                   0.1 * min(df["response_gap_hrs"].iloc[0] / 100, 0.5))

    predicted = prob >= 0.5
    high_intent = bool(df["high_intent_flag"].iloc[0])

    action = "Auto-reply + alert sales team" if predicted else "No action needed"

    return LeadScoreResponse(
        lead_id=f"E-{os.urandom(4).hex().upper()}",
        missed_probability=round(prob, 4),
        predicted_missed=predicted,
        high_intent=high_intent,
        channel=req.channel,
        response_gap_hrs=req.response_gap_hrs,
        recommended_action=action,
    )


@app.post("/api/v1/score/batch", response_model=list, tags=["Scoring"])
async def score_leads_batch(
    req: LeadScoreBatchRequest,
    api_key: str = Depends(verify_api_key),
):
    """Score a batch of leads."""
    results = []
    for lead in req.leads:
        result = await score_lead(lead, api_key)
        results.append(result)
    return results


@app.post("/api/v1/reply/preview", response_model=ReplyPreviewResponse, tags=["Reply"])
async def preview_reply(
    req: ReplyPreviewRequest,
    api_key: str = Depends(verify_api_key),
):
    """Preview a smart reply for a lead without sending."""
    from smart_reply_engine import generate_reply
    reply = generate_reply(
        customer_name=req.customer_name,
        customer_email=req.customer_email,
        subject=req.subject,
        message_text=req.message_text,
        channel=req.channel,
    )
    return ReplyPreviewResponse(
        reply_subject=reply["reply_subject"],
        reply_body=reply["reply_body"],
        detected_intent=reply["detected_intent"],
        intent_scores=reply["intent_scores"],
        channel=reply["channel"],
        is_auto_replied=reply["is_auto_replied"],
        generated_at=reply["generated_at"],
    )


@app.post("/api/v1/reply/send", response_model=ReplySendResponse, tags=["Reply"])
async def send_reply(
    req: ReplySendRequest,
    api_key: str = Depends(verify_api_key),
):
    """Send a follow-up email to a missed lead."""
    from auto_followup import send_followup
    from smart_reply_engine import generate_reply

    # Generate smart reply
    reply = generate_reply(
        customer_name=req.customer_name,
        customer_email=req.customer_email,
        subject=req.subject,
        message_text="",
        channel=req.channel,
    )

    lead_payload = {
        "lead_id": req.lead_id,
        "customer_email": req.customer_email,
        "customer_name": req.customer_name,
        "channel": req.channel,
        "subject": req.subject,
        "original_message_id": req.original_message_id,
        "reply_subject": reply["reply_subject"],
        "reply_body": reply["reply_body"],
        "detected_intent": reply["detected_intent"],
    }

    success = send_followup(lead_payload)

    return ReplySendResponse(
        success=success,
        lead_id=req.lead_id,
        customer_email=req.customer_email,
        message="Reply sent successfully" if success else "Failed to send reply (demo mode or dedup)",
    )


@app.get("/api/v1/leads", response_model=LeadListResponse, tags=["Leads"])
async def list_leads(
    status: Optional[str] = None,
    channel: Optional[str] = None,
    high_intent: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
    api_key: str = Depends(verify_api_key),
):
    """List scored leads with filtering."""
    if not os.path.exists(SCORED_CSV):
        return LeadListResponse(total=0, missed_count=0, leads=[])

    df = pd.read_csv(SCORED_CSV)

    # Apply filters
    if status == "missed":
        df = df[df["predicted_missed"] == 1]
    elif status == "responded":
        df = df[df["predicted_missed"] == 0]

    if channel:
        df = df[df["channel"].str.lower() == channel.lower()]

    if high_intent is not None:
        df = df[df["high_intent_flag"] == (1 if high_intent else 0)]

    total = len(df)
    missed_count = int(df["predicted_missed"].sum()) if "predicted_missed" in df.columns else 0

    # Paginate
    df_page = df.iloc[offset:offset + limit]

    leads = []
    for _, row in df_page.iterrows():
        leads.append(LeadDetailResponse(
            lead_id=row.get("lead_id", ""),
            channel=row.get("channel", ""),
            message_text=str(row.get("message_text", ""))[:200],
            message_hour=int(row.get("message_hour", 0)),
            message_length=int(row.get("message_length", 0)),
            high_intent_flag=int(row.get("high_intent_flag", 0)),
            prev_contacts=int(row.get("prev_contacts", 0)),
            response_gap_hrs=float(row.get("response_gap_hrs", 0)),
            missed_probability=float(row.get("missed_probability", 0)),
            predicted_missed=int(row.get("predicted_missed", 0)),
            customer_name=row.get("_customer_name"),
            customer_email=row.get("_customer_email"),
            subject=row.get("_subject"),
            received_time=row.get("_received_time"),
        ))

    return LeadListResponse(total=total, missed_count=missed_count, leads=leads)


@app.get("/api/v1/leads/{lead_id}", response_model=LeadDetailResponse, tags=["Leads"])
async def get_lead(
    lead_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get a specific lead by ID."""
    if not os.path.exists(SCORED_CSV):
        raise HTTPException(status_code=404, detail="No scored leads found")

    df = pd.read_csv(SCORED_CSV)
    match = df[df["lead_id"] == lead_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} not found")

    row = match.iloc[0]
    return LeadDetailResponse(
        lead_id=row.get("lead_id", ""),
        channel=row.get("channel", ""),
        message_text=str(row.get("message_text", ""))[:500],
        message_hour=int(row.get("message_hour", 0)),
        message_length=int(row.get("message_length", 0)),
        high_intent_flag=int(row.get("high_intent_flag", 0)),
        prev_contacts=int(row.get("prev_contacts", 0)),
        response_gap_hrs=float(row.get("response_gap_hrs", 0)),
        missed_probability=float(row.get("missed_probability", 0)),
        predicted_missed=int(row.get("predicted_missed", 0)),
        customer_name=row.get("_customer_name"),
        customer_email=row.get("_customer_email"),
        subject=row.get("_subject"),
        received_time=row.get("_received_time"),
    )


@app.put("/api/v1/leads/{lead_id}/followup", tags=["Leads"])
async def mark_lead_followed_up(
    lead_id: str,
    req: LeadUpdateRequest,
    api_key: str = Depends(verify_api_key),
):
    """Mark a lead as human-followed-up."""
    import json
    followup_status = _load_json(FOLLOWUP_LOG, {})
    if lead_id not in followup_status:
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} not in follow-up tracking")

    followup_status[lead_id]["human_followed_up"] = req.human_followed_up
    followup_status[lead_id]["human_followed_up_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with open(FOLLOWUP_LOG, "w") as f:
        json.dump(followup_status, f, indent=2, default=str)

    return {"status": "updated", "lead_id": lead_id, "human_followed_up": req.human_followed_up}


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats(
    api_key: str = Depends(verify_api_key),
):
    """Get pipeline statistics."""
    total_leads = 0
    missed_leads = 0

    if os.path.exists(SCORED_CSV):
        df = pd.read_csv(SCORED_CSV)
        total_leads = len(df)
        missed_leads = int(df["predicted_missed"].sum()) if "predicted_missed" in df.columns else 0

    followup_status = _load_json(FOLLOWUP_LOG, {})
    reply_log = _load_json(REPLY_LOG, [])

    auto_replied = len(reply_log) if isinstance(reply_log, list) else 0
    human_followed = sum(
        1 for s in followup_status.values()
        if isinstance(s, dict) and s.get("human_followed_up")
    )
    pending = sum(
        1 for s in followup_status.values()
        if isinstance(s, dict) and s.get("auto_replied") and not s.get("human_followed_up")
    )

    recovery_rate = (len(followup_status) / missed_leads * 100) if missed_leads > 0 else 100.0

    return StatsResponse(
        total_leads=total_leads,
        missed_leads=missed_leads,
        auto_replied=auto_replied,
        human_followed_up=human_followed,
        pending=pending,
        recovery_rate=round(recovery_rate, 1),
        scan_count=len(_load_json(os.path.join(LOG_DIR, "followup_status.json"), {})),
    )


@app.post("/api/v1/scan", response_model=ScanResponse, tags=["Pipeline"])
async def trigger_scan(
    req: ScanRequest = ScanRequest(),
    api_key: str = Depends(verify_api_key),
):
    """Trigger an inbox scan (requires IMAP credentials)."""
    if not os.getenv("IMAP_USER"):
        raise HTTPException(
            status_code=503,
            detail="Gmail not connected. Set IMAP_USER and IMAP_PASS environment variables."
        )

    from email_reader import fetch_customer_emails
    from inbox_monitor import load_artefacts, score_email

    df = fetch_customer_emails(max_emails=req.max_emails, search_since_days=req.search_since_days)
    if df.empty:
        return ScanResponse(
            status="completed",
            scanned=0,
            missed_detected=0,
            replied=0,
            skipped=0,
            message="No new customer emails found",
        )

    ensemble, scaler = load_artefacts()
    if ensemble and scaler:
        df = score_email(df, ensemble, scaler)
    else:
        df["missed_probability"] = 0.5
        df["predicted_missed"] = 0

    missed = int(df["predicted_missed"].sum())
    os.makedirs(os.path.dirname(SCORED_CSV), exist_ok=True)
    df.to_csv(SCORED_CSV, index=False)

    return ScanResponse(
        status="completed",
        scanned=len(df),
        missed_detected=missed,
        replied=0,
        skipped=0,
        message=f"Scan complete: {len(df)} emails scored, {missed} missed leads detected",
    )


# ── Run ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
