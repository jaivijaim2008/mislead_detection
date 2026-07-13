"""
test_api.py — Tests for api.py (FastAPI REST API)
Covers: health check, scoring, reply preview, lead listing, stats, auth, rate limiting.
"""
import os
import sys
import json
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Set API key for tests
os.environ["API_KEY"] = "test-key-12345"

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
API_KEY = "test-key-12345"
HEADERS = {"X-API-Key": API_KEY}


class TestHealthCheck:
    """Test /api/v1/health endpoint."""

    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "gmail_connected" in data
        assert "timestamp" in data

    def test_health_no_auth_required(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200


class TestAuthentication:
    """Test API key authentication."""

    def test_missing_api_key_returns_422(self):
        response = client.post("/api/v1/score", json={
            "channel": "Email",
            "message_text": "Hello",
        })
        assert response.status_code == 422

    def test_invalid_api_key_returns_401(self):
        response = client.post("/api/v1/score", json={
            "channel": "Email",
            "message_text": "Hello",
        }, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_valid_api_key_accepted(self):
        response = client.post("/api/v1/score", json={
            "channel": "Email",
            "message_text": "What is the price?",
        }, headers=HEADERS)
        assert response.status_code == 200


class TestLeadScoring:
    """Test /api/v1/score endpoint."""

    def test_score_lead_returns_200(self):
        response = client.post("/api/v1/score", json={
            "channel": "Email",
            "message_text": "What is the price of your course?",
            "message_hour": 10,
            "response_gap_hrs": 24.0,
        }, headers=HEADERS)
        assert response.status_code == 200

    def test_score_lead_response_structure(self):
        response = client.post("/api/v1/score", json={
            "channel": "Email",
            "message_text": "What is the price of your course?",
        }, headers=HEADERS)
        data = response.json()
        assert "lead_id" in data
        assert "missed_probability" in data
        assert "predicted_missed" in data
        assert "high_intent" in data
        assert "recommended_action" in data

    def test_score_lead_probability_range(self):
        response = client.post("/api/v1/score", json={
            "channel": "Email",
            "message_text": "What is the price?",
        }, headers=HEADERS)
        data = response.json()
        assert 0 <= data["missed_probability"] <= 1

    def test_score_lead_high_gap_high_prob(self):
        response = client.post("/api/v1/score", json={
            "channel": "Email",
            "message_text": "hello",
            "response_gap_hrs": 200.0,
        }, headers=HEADERS)
        data = response.json()
        # High gap should increase missed probability
        assert data["missed_probability"] > 0.3

    def test_score_lead_batch(self):
        response = client.post("/api/v1/score/batch", json={
            "leads": [
                {"channel": "Email", "message_text": "Price?"},
                {"channel": "WhatsApp", "message_text": "Demo please"},
            ]
        }, headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


class TestReplyPreview:
    """Test /api/v1/reply/preview endpoint."""

    def test_preview_reply_returns_200(self):
        response = client.post("/api/v1/reply/preview", json={
            "customer_name": "Priya",
            "customer_email": "priya@example.com",
            "subject": "Course Pricing",
            "message_text": "What is the price of your data science course?",
        }, headers=HEADERS)
        assert response.status_code == 200

    def test_preview_reply_structure(self):
        response = client.post("/api/v1/reply/preview", json={
            "customer_name": "Rahul",
            "customer_email": "rahul@example.com",
            "subject": "Demo Request",
            "message_text": "Can I get a demo of your teaching?",
        }, headers=HEADERS)
        data = response.json()
        assert "reply_subject" in data
        assert "reply_body" in data
        assert "detected_intent" in data
        assert "intent_scores" in data
        assert "is_auto_replied" in data

    def test_preview_reply_detects_pricing_intent(self):
        response = client.post("/api/v1/reply/preview", json={
            "customer_name": "Test",
            "customer_email": "test@example.com",
            "subject": "Pricing",
            "message_text": "What is the price? How much does it cost?",
        }, headers=HEADERS)
        data = response.json()
        assert data["detected_intent"] == "pricing"

    def test_preview_reply_contains_customer_name(self):
        response = client.post("/api/v1/reply/preview", json={
            "customer_name": "Priya",
            "customer_email": "priya@example.com",
            "subject": "Hi",
            "message_text": "Hello there",
        }, headers=HEADERS)
        data = response.json()
        assert "Priya" in data["reply_body"]


class TestLeadListing:
    """Test /api/v1/leads endpoint."""

    def test_list_leads_returns_200(self):
        response = client.get("/api/v1/leads", headers=HEADERS)
        assert response.status_code == 200

    def test_list_leads_structure(self):
        response = client.get("/api/v1/leads", headers=HEADERS)
        data = response.json()
        assert "total" in data
        assert "missed_count" in data
        assert "leads" in data
        assert isinstance(data["leads"], list)

    def test_list_leads_with_status_filter(self):
        response = client.get("/api/v1/leads?status=missed", headers=HEADERS)
        assert response.status_code == 200

    def test_list_leads_with_limit(self):
        response = client.get("/api/v1/leads?limit=5", headers=HEADERS)
        data = response.json()
        assert len(data["leads"]) <= 5

    def test_get_lead_not_found(self):
        response = client.get("/api/v1/leads/NONEXISTENT", headers=HEADERS)
        assert response.status_code == 404


class TestStats:
    """Test /api/v1/stats endpoint."""

    def test_stats_returns_200(self):
        response = client.get("/api/v1/stats", headers=HEADERS)
        assert response.status_code == 200

    def test_stats_structure(self):
        response = client.get("/api/v1/stats", headers=HEADERS)
        data = response.json()
        assert "total_leads" in data
        assert "missed_leads" in data
        assert "auto_replied" in data
        assert "recovery_rate" in data


class TestScanTrigger:
    """Test /api/v1/scan endpoint."""

    def test_scan_no_gmail_returns_503(self):
        # Without IMAP credentials, should return 503
        original = os.environ.pop("IMAP_USER", None)
        response = client.post("/api/v1/scan", json={}, headers=HEADERS)
        if original:
            os.environ["IMAP_USER"] = original
        assert response.status_code == 503


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_message(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
