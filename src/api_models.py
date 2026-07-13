"""
api_models.py — Missed-Lead Detector
Pydantic models for REST API request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Request Models ──────────────────────────────────────────

class LeadScoreRequest(BaseModel):
    """Request to score a single lead."""
    channel: str = Field(..., description="Communication channel", example="Email")
    message_text: str = Field(..., description="Customer message text", example="Hi, what is the price?")
    message_hour: int = Field(12, ge=0, le=23, description="Hour of day (0-23)")
    message_length: Optional[int] = Field(None, description="Message character count")
    high_intent_flag: Optional[int] = Field(None, ge=0, le=1, description="1 if high-intent keywords present")
    prev_contacts: int = Field(0, ge=0, description="Number of prior contacts")
    response_gap_hrs: float = Field(24.0, ge=0, description="Hours since customer inquiry")


class LeadScoreBatchRequest(BaseModel):
    """Request to score a batch of leads."""
    leads: List[LeadScoreRequest]


class ReplyPreviewRequest(BaseModel):
    """Request to preview a smart reply."""
    customer_name: str = Field(..., description="Customer name", example="Priya")
    customer_email: str = Field(..., description="Customer email", example="priya@example.com")
    subject: str = Field(..., description="Email subject", example="Course Pricing")
    message_text: str = Field(..., description="Customer message text", example="What is the price?")
    channel: str = Field("Email", description="Communication channel")


class ReplySendRequest(BaseModel):
    """Request to send a follow-up email."""
    lead_id: str = Field(..., description="Lead ID", example="E-A1B2C3D4")
    customer_email: str = Field(..., description="Recipient email", example="priya@example.com")
    customer_name: str = Field(..., description="Customer name", example="Priya")
    channel: str = Field("Email", description="Channel")
    subject: str = Field(..., description="Original subject", example="Course Pricing")
    original_message_id: Optional[str] = Field(None, description="For threading")


class ScanRequest(BaseModel):
    """Request to trigger an inbox scan."""
    max_emails: int = Field(30, ge=1, le=100, description="Max emails to scan")
    search_since_days: int = Field(7, ge=1, le=90, description="Look back N days")
    dry_run: bool = Field(False, description="Preview only, don't send")


class LeadUpdateRequest(BaseModel):
    """Request to mark a lead as human-followed-up."""
    human_followed_up: bool = Field(True, description="Mark as human-followed-up")


# ── Response Models ─────────────────────────────────────────

class LeadScoreResponse(BaseModel):
    """Response for lead scoring."""
    lead_id: str
    missed_probability: float
    predicted_missed: bool
    high_intent: bool
    channel: str
    response_gap_hrs: float
    recommended_action: str


class ReplyPreviewResponse(BaseModel):
    """Response for reply preview."""
    reply_subject: str
    reply_body: str
    detected_intent: str
    intent_scores: Dict[str, float]
    channel: str
    is_auto_replied: bool
    generated_at: str


class ReplySendResponse(BaseModel):
    """Response for sending a reply."""
    success: bool
    lead_id: str
    customer_email: str
    message: str


class LeadDetailResponse(BaseModel):
    """Detailed lead information."""
    lead_id: str
    channel: str
    message_text: str
    message_hour: int
    message_length: int
    high_intent_flag: int
    prev_contacts: int
    response_gap_hrs: float
    missed_probability: float
    predicted_missed: int
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    subject: Optional[str] = None
    received_time: Optional[str] = None


class LeadListResponse(BaseModel):
    """Response for listing leads."""
    total: int
    missed_count: int
    leads: List[LeadDetailResponse]


class StatsResponse(BaseModel):
    """Pipeline statistics."""
    total_leads: int
    missed_leads: int
    auto_replied: int
    human_followed_up: int
    pending: int
    recovery_rate: float
    scan_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
    gmail_connected: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class ScanResponse(BaseModel):
    """Response for inbox scan trigger."""
    status: str
    scanned: int
    missed_detected: int
    replied: int
    skipped: int
    message: str
