"""
test_train_model.py — Tests for train_model.py
Covers: feature engineering, clustering, model training pipeline.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestFeatureEngineering:
    """Test load_and_engineer function."""

    def test_load_and_engineer_returns_tuple(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(df, pd.DataFrame)

    def test_features_correct_count(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        assert X.shape[1] == 9  # 9 features

    def test_feature_names(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        expected_features = [
            "channel_enc", "message_length", "high_intent_flag", "prev_contacts",
            "response_gap_hrs", "intent_score", "is_business_hours",
            "gap_bucket", "message_hour"
        ]
        assert list(X.columns) == expected_features

    def test_target_encoding(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        # replied=1 → y=0 (replied), replied=0 → y=1 (missed)
        assert set(y.unique()).issubset({0, 1})

    def test_channel_encoding(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        assert X["channel_enc"].dtype in [np.int64, np.int32, int]

    def test_intent_score_computed(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        assert (X["intent_score"] >= 0).all()

    def test_business_hours_flag(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        assert set(X["is_business_hours"].unique()).issubset({0, 1})

    def test_gap_bucket_values(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        assert set(X["gap_bucket"].unique()).issubset({0, 1, 2, 3})


class TestClustering:
    """Test cluster_leads function."""

    def test_cluster_leads_adds_columns(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer, cluster_leads
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        df = cluster_leads(df)
        assert "cluster" in df.columns
        assert "segment" in df.columns

    def test_cluster_leads_has_three_clusters(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer, cluster_leads
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        df = cluster_leads(df)
        assert df["cluster"].nunique() == 3

    def test_segment_names(self, sample_leads_df, tmp_path):
        from train_model import load_and_engineer, cluster_leads
        csv_path = tmp_path / "test_leads.csv"
        sample_leads_df.to_csv(csv_path, index=False)
        X, y, df = load_and_engineer(str(csv_path))
        df = cluster_leads(df)
        valid_segments = {"High-Intent-Missed", "Low-Intent", "Already-Converted"}
        assert set(df["segment"].unique()).issubset(valid_segments)
