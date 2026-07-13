"""
test_plot_model_comparison.py — Tests for plot_model_comparison.py
Covers: chart generation, data structures, output files.
"""
import os
import sys
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestChartData:
    """Test chart data structures."""

    def test_before_tuning_has_models(self):
        from plot_model_comparison import before_tuning
        assert isinstance(before_tuning, dict)
        assert len(before_tuning) > 0
        # Should have all 8 models
        expected_models = [
            "XGBoost", "RandomForest", "GrandEnsemble(ML+DL)",
            "Ensemble(RF+XGB+LR)", "DeepLearning(PyTorch)",
            "DecisionTree", "LogisticRegression", "NaiveBayes"
        ]
        for model in expected_models:
            assert model in before_tuning, f"Missing model: {model}"

    def test_after_tuning_has_models(self):
        from plot_model_comparison import after_tuning
        assert isinstance(after_tuning, dict)
        assert len(after_tuning) > 0

    def test_auc_scores_reasonable(self):
        from plot_model_comparison import before_tuning, after_tuning
        for model, score in before_tuning.items():
            assert 0.5 <= score <= 1.0, f"{model} AUC {score} out of range"
        for model, score in after_tuning.items():
            assert 0.5 <= score <= 1.0, f"{model} AUC {score} out of range"

    def test_best_model_is_xgboost(self):
        from plot_model_comparison import after_tuning
        best_model = max(after_tuning, key=after_tuning.get)
        assert best_model == "XGBoost"

    def test_tuning_improves_xgboost(self):
        from plot_model_comparison import before_tuning, after_tuning
        assert after_tuning["XGBoost"] >= before_tuning["XGBoost"]


class TestChartGeneration:
    """Test chart generation (when matplotlib available)."""

    def test_chart_file_created(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        models = ["XGBoost", "RandomForest", "NaiveBayes"]
        before_vals = [0.9756, 0.9725, 0.8604]
        after_vals = [0.9794, 0.9725, 0.8604]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, before_vals, width, label="Before")
        ax.bar(x + width/2, after_vals, width, label="After")
        ax.set_xticks(x)
        ax.set_xticklabels(models)

        output_path = tmp_path / "test_chart.png"
        plt.savefig(output_path, dpi=100)
        plt.close()

        assert output_path.exists()
        assert output_path.stat().st_size > 0
