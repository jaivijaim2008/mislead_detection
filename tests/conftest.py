"""
conftest.py — Shared pytest fixtures for Missed-Lead Detector tests.
"""
import os
import sys
import json
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def sample_leads_df():
    """Sample DataFrame matching the unified training schema."""
    return pd.DataFrame({
        "lead_id": ["L00001", "L00002", "L00003", "L00004", "L00005"],
        "channel": ["Email", "WhatsApp", "Phone Inquiry", "Website Chat", "Email"],
        "message_text": [
            "Hi, I want to know the price of your data science course. Do you have EMI options?",
            "Can I get a demo of your teaching before I enroll?",
            "Do you provide placement assistance after the course?",
            "Hello, just checking what you offer.",
            "URGENT: I need immediate response. My enrollment is pending!",
        ],
        "message_hour": [10, 14, 9, 21, 11],
        "message_length": [85, 48, 51, 35, 55],
        "high_intent_flag": [1, 1, 1, 0, 1],
        "prev_contacts": [0, 2, 1, 0, 3],
        "response_gap_hrs": [48.5, 12.3, 72.0, 120.0, 6.0],
        "replied": [0, 1, 0, 0, 1],
    })


@pytest.fixture
def sample_lead_dict():
    """Sample lead dict as used by auto_followup and smart_reply_engine."""
    return {
        "lead_id": "L00042",
        "customer_email": "priya@example.com",
        "customer_name": "Priya",
        "channel": "Email",
        "subject": "Course Pricing",
        "original_message_id": "<abc123@mail.example.com>",
        "message_text": "Hi, I want to know the price of your data science course.",
        "reply_subject": "Re: Course Pricing — Pricing Details",
        "reply_body": "Hi Priya,\n\nThank you for your interest!\n\nBest regards,\nSales Team",
        "detected_intent": "pricing",
    }


@pytest.fixture
def sample_email_row():
    """Single email row as returned by email_reader.fetch_customer_emails."""
    return {
        "lead_id": "E-A1B2C3D4",
        "channel": "Email",
        "message_text": "Hi, I want to know the price of your data science course.",
        "message_hour": 10,
        "message_length": 55,
        "high_intent_flag": 1,
        "prev_contacts": 0,
        "response_gap_hrs": 24.5,
        "_customer_email": "priya@gmail.com",
        "_customer_name": "Priya Sharma",
        "_subject": "Course Pricing",
        "_message_id": "<test123@mail.gmail.com>",
        "_received_time": "2025-01-15 10:00 UTC",
    }


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs, clean up after."""
    d = tempfile.mkdtemp(prefix="mld_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_models_dir(tmp_dir):
    """Create a temporary models directory."""
    d = os.path.join(tmp_dir, "models")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture
def tmp_outputs_dir(tmp_dir):
    """Create a temporary outputs directory."""
    d = os.path.join(tmp_dir, "outputs")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture
def tmp_logs_dir(tmp_dir):
    """Create a temporary logs directory."""
    d = os.path.join(tmp_dir, "logs")
    os.makedirs(d, exist_ok=True)
    return d
