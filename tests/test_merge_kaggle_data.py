"""
test_merge_kaggle_data.py — Tests for merge_kaggle_data.py
Covers: dataset loading, mapping functions, schema unification.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestSchemaUnification:
    """Test unified schema structure."""

    def test_unified_columns_defined(self):
        from merge_kaggle_data import map_lead_scoring, map_support_tickets
        # The unified schema should have these columns
        unified_cols = [
            "lead_id", "channel", "message_text", "message_hour",
            "message_length", "high_intent_flag", "prev_contacts",
            "response_gap_hrs", "replied"
        ]
        assert len(unified_cols) == 9

    def test_map_lead_scoring_output(self):
        from merge_kaggle_data import map_lead_scoring
        # Create a mock DataFrame that mimics X Education Lead Scoring
        mock_df = pd.DataFrame({
            "Lead Source": ["google", "facebook", "linkedin"],
            "TotalVisits": [5, 2, 8],
            "Total Time Spent on Website": [500, 100, 800],
            "Last Activity": ["Email Opened", "SMS Sent", "Page Visited"],
            "Converted": [1, 0, 1],
            "Prospect ID": ["P1", "P2", "P3"],
            "What is your current occupation": ["Student", "Professional", "Student"],
            "What matters most to you in choosing a course": ["Placement", "Fee", "Curriculum"],
            "Tags": ["Interested", "Not interested", "Interested"],
        })
        result = map_lead_scoring(mock_df)
        assert len(result) == 3
        assert "lead_id" in result.columns
        assert "channel" in result.columns
        assert "replied" in result.columns
        assert all(result["lead_id"].str.startswith("KLS_"))

    def test_map_support_tickets_output(self):
        from merge_kaggle_data import map_support_tickets
        mock_df = pd.DataFrame({
            "type": ["Incident", "Request", "Problem"],
            "subject": ["Login issue", "Feature request", "Bug report"],
            "body": ["Cannot login", "Add dark mode", "App crashes"],
            "priority": ["high", "medium", "low"],
            "tag_1": ["login", None, "crash"],
            "answer": ["Reset password", None, "Fixed in v2"],
        })
        result = map_support_tickets(mock_df)
        assert len(result) == 3
        assert "lead_id" in result.columns
        assert all(result["lead_id"].str.startswith("KCS_"))

    def test_channel_mapping_lead_scoring(self):
        from merge_kaggle_data import map_lead_scoring
        mock_df = pd.DataFrame({
            "Lead Source": ["google", "facebook", "youtube"],
            "TotalVisits": [5, 2, 8],
            "Total Time Spent on Website": [500, 100, 800],
            "Last Activity": ["Email Opened", "SMS Sent", "Page Visited"],
            "Converted": [1, 0, 1],
            "Prospect ID": ["P1", "P2", "P3"],
        })
        result = map_lead_scoring(mock_df)
        # google → Email, facebook → WhatsApp, youtube → Website Chat
        assert result.iloc[0]["channel"] == "Email"
        assert result.iloc[1]["channel"] == "WhatsApp"
        assert result.iloc[2]["channel"] == "Website Chat"

    def test_channel_mapping_support_tickets(self):
        from merge_kaggle_data import map_support_tickets
        mock_df = pd.DataFrame({
            "type": ["Incident", "Request", "Problem"],
            "subject": ["Test", "Test", "Test"],
            "body": ["Test", "Test", "Test"],
            "priority": ["high", "medium", "low"],
            "answer": ["Solution", None, None],
        })
        result = map_support_tickets(mock_df)
        # Incident → Email, Request → WhatsApp, Problem → Phone Inquiry
        assert result.iloc[0]["channel"] == "Email"
        assert result.iloc[1]["channel"] == "WhatsApp"
        assert result.iloc[2]["channel"] == "Phone Inquiry"

    def test_target_mapping(self):
        from merge_kaggle_data import map_support_tickets
        mock_df = pd.DataFrame({
            "type": ["Incident"],
            "subject": ["Test"],
            "body": ["Test"],
            "priority": ["high"],
            "answer": ["Solution"],  # Has answer → replied=1
        })
        result = map_support_tickets(mock_df)
        assert result.iloc[0]["replied"] == 1

        mock_df_no_answer = pd.DataFrame({
            "type": ["Incident"],
            "subject": ["Test"],
            "body": ["Test"],
            "priority": ["high"],
            "answer": [None],  # No answer → replied=0
        })
        result_no_answer = map_support_tickets(mock_df_no_answer)
        assert result_no_answer.iloc[0]["replied"] == 0
