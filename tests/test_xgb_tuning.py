"""
test_xgb_tuning.py — Tests for xgb_tuning.py
Covers: Optuna objective function, tuning pipeline.
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestObjectiveFunction:
    """Test create_objective function."""

    def test_objective_returns_callable(self):
        from xgb_tuning import create_objective
        X = np.random.randn(100, 9).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        obj = create_objective(X, y)
        assert callable(obj)

    def test_objective_accepts_trial(self):
        from xgb_tuning import create_objective
        import optuna
        X = np.random.randn(100, 9).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        obj = create_objective(X, y)

        study = optuna.create_study(direction="maximize")
        # Run just 1 trial to test
        study.optimize(obj, n_trials=1, show_progress_bar=False)
        assert len(study.trials) == 1
        assert study.best_value is not None
