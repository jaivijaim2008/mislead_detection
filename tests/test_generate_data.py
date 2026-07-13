"""
test_generate_data.py — Tests for generate_data.py
Covers: synthetic data generation, schema validation, distributions.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestGenerateLeads:
    """Test the generate_leads function."""

    def test_generates_correct_number_of_leads(self):
        from generate_data import generate_leads
        df = generate_leads(n=100)
        assert len(df) == 100

    def test_generates_5000_leads(self):
        from generate_data import generate_leads
        df = generate_leads(n=5000)
        assert len(df) == 5000

    def test_schema_has_required_columns(self):
        from generate_data import generate_leads
        df = generate_leads(n=10)
        required_cols = [
            "lead_id", "channel", "message_text", "message_hour",
            "message_length", "high_intent_flag", "prev_contacts",
            "response_gap_hrs", "replied"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_lead_id_format(self):
        from generate_data import generate_leads
        df = generate_leads(n=50)
        for lead_id in df["lead_id"]:
            assert lead_id.startswith("L")
            assert len(lead_id) == 6  # L + 5 digits

    def test_lead_id_unique(self):
        from generate_data import generate_leads
        df = generate_leads(n=200)
        assert df["lead_id"].nunique() == len(df)

    def test_channel_values_valid(self):
        from generate_data import generate_leads, CHANNELS
        df = generate_leads(n=200)
        valid_channels = set(CHANNELS)
        for ch in df["channel"]:
            assert ch in valid_channels, f"Invalid channel: {ch}"

    def test_message_hour_range(self):
        from generate_data import generate_leads
        df = generate_leads(n=200)
        assert df["message_hour"].min() >= 0
        assert df["message_hour"].max() <= 23

    def test_message_length_positive(self):
        from generate_data import generate_leads
        df = generate_leads(n=100)
        assert (df["message_length"] > 0).all()

    def test_high_intent_flag_binary(self):
        from generate_data import generate_leads
        df = generate_leads(n=200)
        unique_flags = df["high_intent_flag"].unique()
        assert set(unique_flags).issubset({0, 1})

    def test_prev_contacts_non_negative(self):
        from generate_data import generate_leads
        df = generate_leads(n=200)
        assert (df["prev_contacts"] >= 0).all()

    def test_response_gap_positive(self):
        from generate_data import generate_leads
        df = generate_leads(n=200)
        assert (df["response_gap_hrs"] > 0).all()

    def test_replied_binary(self):
        from generate_data import generate_leads
        df = generate_leads(n=200)
        unique_replied = df["replied"].unique()
        assert set(unique_replied).issubset({0, 1})

    def test_target_distribution_reasonable(self):
        from generate_data import generate_leads
        df = generate_leads(n=1000)
        replied_rate = df["replied"].mean()
        # Should have a mix of replied and missed
        assert 0.2 < replied_rate < 0.8, f"Unusual replied rate: {replied_rate}"

    def test_message_text_not_empty(self):
        from generate_data import generate_leads
        df = generate_leads(n=50)
        for text in df["message_text"]:
            assert isinstance(text, str)
            assert len(text) > 0

    def test_reproducibility_with_seed(self):
        import random
        import numpy as np
        from generate_data import generate_leads
        # Reset seeds before each call to ensure reproducibility
        random.seed(42)
        np.random.seed(42)
        df1 = generate_leads(n=20)
        random.seed(42)
        np.random.seed(42)
        df2 = generate_leads(n=20)
        pd.testing.assert_frame_equal(df1, df2)
