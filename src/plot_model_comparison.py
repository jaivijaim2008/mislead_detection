"""
plot_model_comparison.py — Missed-Lead Detector
Generates a before/after hyperparameter tuning AUC comparison bar chart.

"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "..", "outputs")

# ── AUC scores before Optuna tuning (from first merged-dataset run) ──
before_tuning = {
    "XGBoost":              0.9756,
    "RandomForest":         0.9725,
    "GrandEnsemble(ML+DL)": 0.9726,
    "Ensemble(RF+XGB+LR)":  0.9715,
    "DeepLearning(PyTorch)": 0.9655,
    "DecisionTree":         0.9627,
    "LogisticRegression":   0.9132,
    "NaiveBayes":           0.8604,
}

# ── AUC scores after Optuna tuning ──
after_tuning = {
    "XGBoost":              0.9794,
    "RandomForest":         0.9725,
    "GrandEnsemble(ML+DL)": 0.9709,
    "Ensemble(RF+XGB+LR)":  0.9720,
    "DeepLearning(PyTorch)": 0.9613,
    "DecisionTree":         0.9627,
    "LogisticRegression":   0.9132,
    "NaiveBayes":           0.8604,
}

# Sort by after-tuning AUC (descending)
models = sorted(after_tuning.keys(), key=lambda m: after_tuning[m], reverse=True)
before_vals = [before_tuning[m] for m in models]
after_vals = [after_tuning[m] for m in models]
deltas = [after_tuning[m] - before_tuning[m] for m in models]

# ── Create the chart ──
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.35

# Bars
bars_before = ax.bar(x - width/2, before_vals, width, label="Before Tuning",
                      color="#90CAF9", edgecolor="#1565C0", linewidth=0.8, zorder=3)
bars_after = ax.bar(x + width/2, after_vals, width, label="After Tuning (Optuna)",
                     color="#1B5E20", edgecolor="#0D3B0F", linewidth=0.8, zorder=3)

# AUC value labels on top of each bar
for bar, val in zip(bars_before, before_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#1565C0",
            fontweight="bold")

for bar, val in zip(bars_after, after_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#1B5E20",
            fontweight="bold")

# Delta annotations (arrows showing improvement)
for i, (m, delta) in enumerate(zip(models, deltas)):
    if delta > 0.0001:
        ax.annotate(f"+{delta:.4f}", xy=(x[i] + width/2, after_vals[i] + 0.006),
                    fontsize=7, color="#E65100", ha="center", fontweight="bold")
    elif delta < -0.0001:
        ax.annotate(f"{delta:.4f}", xy=(x[i] + width/2, after_vals[i] + 0.006),
                    fontsize=7, color="#B71C1C", ha="center", fontweight="bold")

# Styling
ax.set_xlabel("Model", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("AUC Score", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title("Model Performance: Before vs After Optuna Hyperparameter Tuning\n"
             "(13,740 merged samples — X Education + Customer Support + Synthetic)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
ax.set_ylim(0.84, 0.995)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

# Highlight XGBoost (best model) with a star
best_idx = models.index("XGBoost")
ax.plot(x[best_idx], after_vals[best_idx] + 0.009, marker="*", markersize=15,
        color="#FFD600", zorder=5)
ax.text(x[best_idx], after_vals[best_idx] + 0.011, "BEST",
        ha="center", va="bottom", fontsize=8, fontweight="bold", color="#FF6F00")

plt.tight_layout()

# Save
output_path = os.path.join(OUT, "model_comparison_chart.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[plot] Saved model comparison chart -> {output_path}")

# ── Also save a summary table as text ──
summary_lines = []
summary_lines.append(f"{'Model':<28s} {'Before':>8s} {'After':>8s} {'Delta':>8s}")
summary_lines.append("-" * 55)
for m in models:
    delta = after_tuning[m] - before_tuning[m]
    sign = "+" if delta > 0 else ""
    summary_lines.append(
        f"{m:<28s} {before_tuning[m]:>8.4f} {after_tuning[m]:>8.4f} {sign}{delta:>7.4f}"
    )

summary = "\n".join(summary_lines)
print(f"\n{summary}\n")

summary_path = os.path.join(OUT, "model_comparison_table.txt")
with open(summary_path, "w") as f:
    f.write(summary)
print(f"[plot] Saved summary table -> {summary_path}")
