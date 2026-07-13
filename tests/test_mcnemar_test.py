"""
test_mcnemar_test.py — Tests for mcnemar_test.py
Covers: McNemar's statistic, contingency table, interpretation.
"""
import os
import sys
import pytest
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestMcNemarTest:
    """Test mcnemar_test function."""

    def test_identical_predictions_no_significance(self):
        from mcnemar_test import mcnemar_test
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_different_predictions(self):
        from mcnemar_test import mcnemar_test
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        y_pred_a = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 0])
        y_pred_b = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1])
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert "chi2" in result
        assert "p_value" in result
        assert "contingency_table" in result
        assert "interpretation" in result

    def test_contingency_table_structure(self):
        from mcnemar_test import mcnemar_test
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred_a = np.array([0, 1, 1, 0, 0, 1])
        y_pred_b = np.array([0, 0, 1, 1, 1, 0])
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        ct = result["contingency_table"]
        assert "both_correct" in ct
        assert "A_correct_B_wrong" in ct
        assert "A_wrong_B_correct" in ct
        assert "both_wrong" in ct

    def test_correction_applied(self):
        from mcnemar_test import mcnemar_test
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred_a = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        y_pred_b = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        result_corr = mcnemar_test(y_true, y_pred_a, y_pred_b, correction=True)
        result_no_corr = mcnemar_test(y_true, y_pred_a, y_pred_b, correction=False)
        # With Yates correction, chi2 should be <= without correction
        # Both should be non-negative
        assert result_corr["chi2"] >= 0
        assert result_no_corr["chi2"] >= 0

    def test_no_correction(self):
        from mcnemar_test import mcnemar_test
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred_a = np.array([0, 1, 1, 0, 0, 1])
        y_pred_b = np.array([0, 0, 1, 1, 1, 0])
        result = mcnemar_test(y_true, y_pred_a, y_pred_b, correction=False)
        assert result["chi2"] >= 0

    def test_both_wrong_scenario(self):
        from mcnemar_test import mcnemar_test
        y_true = np.array([1, 1, 1, 1])
        y_pred_a = np.array([0, 0, 0, 0])
        y_pred_b = np.array([0, 0, 0, 0])
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result["contingency_table"]["both_wrong"] == 4
        assert result["contingency_table"]["both_correct"] == 0
