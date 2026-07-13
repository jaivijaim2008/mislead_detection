"""
test_deep_learning.py — Tests for deep_learning.py
Covers: model architecture, forward pass, save/load, training loop.
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestModelArchitecture:
    """Test LeadClassifier model."""

    def test_model_instantiation(self):
        from deep_learning import LeadClassifier
        model = LeadClassifier(input_dim=9, dropout=0.3)
        assert model is not None

    def test_model_forward_pass(self):
        import torch
        from deep_learning import LeadClassifier
        model = LeadClassifier(input_dim=9, dropout=0.3)
        x = torch.randn(4, 9)
        output = model(x)
        assert output.shape == (4,)
        assert output.min() >= 0
        assert output.max() <= 1

    def test_model_parameters_exist(self):
        from deep_learning import LeadClassifier
        model = LeadClassifier(input_dim=9, dropout=0.3)
        params = list(model.parameters())
        assert len(params) > 0

    def test_predict_proba_np(self):
        import torch
        from deep_learning import LeadClassifier, DEVICE
        model = LeadClassifier(input_dim=9, dropout=0.3).to(DEVICE)
        model.eval()
        x = np.random.randn(3, 9).astype(np.float32)
        probs = model.predict_proba_np(x)
        assert probs.shape == (3, 2)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        # Each row should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_np(self):
        import torch
        from deep_learning import LeadClassifier, DEVICE
        model = LeadClassifier(input_dim=9, dropout=0.3).to(DEVICE)
        model.eval()
        x = np.random.randn(5, 9).astype(np.float32)
        preds = model.predict_np(x)
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1})


class TestClassWeights:
    """Test compute_class_weights."""

    def test_class_weights_balanced(self):
        from deep_learning import compute_class_weights
        import torch
        y = np.array([0, 0, 0, 1, 1])
        weights = compute_class_weights(y)
        assert weights.shape == (2,)
        # compute_class_weights returns [weight_pos, weight_neg]
        # Minority class (1, 2 samples) gets HIGHER weight than majority (0, 3 samples)
        # weight_pos = 5/(2*2)=1.25, weight_neg = 5/(2*3)=0.833
        assert weights[0].item() > weights[1].item()  # weight_pos > weight_neg


class TestSaveLoad:
    """Test model save/load."""

    def test_save_and_load_model(self, tmp_path):
        import torch
        import pickle
        from sklearn.preprocessing import StandardScaler
        from deep_learning import LeadClassifier, save_dl_model, load_dl_model, DEVICE
        model = LeadClassifier(input_dim=9, dropout=0.3).to(DEVICE)
        # Use a real sklearn scaler (MagicMock can't be pickled)
        scaler = StandardScaler()
        scaler.fit(np.random.randn(50, 9))
        history = {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}

        save_dl_model(model, scaler, history, str(tmp_path))

        # Check files exist
        assert (tmp_path / "dl_model.pt").exists()
        assert (tmp_path / "dl_scaler.pkl").exists()
        assert (tmp_path / "dl_history.pkl").exists()

        # Load and verify
        loaded_model, loaded_scaler = load_dl_model(
            str(tmp_path / "dl_model.pt"),
            str(tmp_path / "dl_scaler.pkl")
        )
        assert loaded_model is not None
        assert isinstance(loaded_scaler, StandardScaler)


class TestTrainingLoop:
    """Test train_deep_model function."""

    def test_train_deep_model_runs(self):
        from deep_learning import train_deep_model
        X_train = np.random.randn(50, 9).astype(np.float32)
        y_train = np.random.randint(0, 2, 50).astype(np.float32)
        X_val = np.random.randn(10, 9).astype(np.float32)
        y_val = np.random.randint(0, 2, 10).astype(np.float32)

        model, history = train_deep_model(
            X_train, y_train, X_val, y_val,
            epochs=5, lr=1e-3, batch_size=16,
            dropout=0.3, patience=3, verbose=False
        )
        assert model is not None
        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_auc" in history
        assert len(history["train_loss"]) > 0
