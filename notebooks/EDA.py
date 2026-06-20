"""
EDA.py — Missed-Lead Detector
Exploratory Data Analysis script that generates comprehensive visualizations
and statistical summaries of the lead data.

Run: python notebooks/EDA.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

DATA    = os.path.join(BASE, "data", "leads.csv")
SCORED  = os.path.join(BASE, "outputs", "leads_scored.csv")
OUT_DIR = os.path.join(BASE, "outputs", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

print("=" * 60)
print("  MISSED-LEAD DETECTOR - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── 1. Load Data ────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"\n[EDA] Loaded {len(df)} leads from {DATA}")
print(f"[EDA] Columns: {list(df.columns)}")
print(f"[EDA] Data types:\n{df.dtypes}\n")

# ── 2. Summary Statistics ───────────────────────────────────
print("[EDA] Descriptive Statistics:")
print(df.describe().to_string())
print()

# ── 3. Missing Values ──────────────────────────────────────
print(f"[EDA] Missing values per column:\n{df.isnull().sum()}")
print()

# ── Compute Derived Features ──────────────────────────────
df["is_business_hours"] = df["message_hour"].between(9, 18).astype(int)

# Compute intent score
intent_words = ["price", "buy", "interested", "demo", "quote", "available"]
df["intent_score"] = df["message_text"].apply(
    lambda t: sum(w in str(t).lower() for w in intent_words))

# ── 4. Target Variable Analysis ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Missed-Lead Detector — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# 4a. Target distribution
ax = axes[0, 0]
target_counts = df["replied"].map({0: "Missed", 1: "Replied"}).value_counts()
colors_target = ["#e74c3c", "#27ae60"]
bars = ax.bar(target_counts.index, target_counts.values, color=colors_target, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, target_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val} ({val/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Target Distribution: Missed vs Replied", fontsize=12, fontweight="bold")
ax.set_ylabel("Count")

# 4b. Channel distribution
ax = axes[0, 1]
channel_counts = df["channel"].value_counts()
channel_colors = ["#3498db", "#9b59b6", "#1abc9c", "#f39c12"]
bars = ax.bar(channel_counts.index, channel_counts.values, color=channel_colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, channel_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val} ({val/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Leads by Channel", fontsize=12, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)

# 4c. Response Gap Distribution
ax = axes[0, 2]
missed_gaps = df[df["replied"] == 0]["response_gap_hrs"]
replied_gaps = df[df["replied"] == 1]["response_gap_hrs"]
ax.hist([replied_gaps, missed_gaps], bins=40, alpha=0.7,
        label=["Replied", "Missed"], color=["#27ae60", "#e74c3c"], edgecolor="white")
ax.set_xlabel("Response Gap (hours)")
ax.set_ylabel("Frequency")
ax.set_title("Response Gap: Replied vs Missed", fontsize=12, fontweight="bold")
ax.legend()
ax.set_xlim(0, 200)

# 4d. Missed rate by channel
ax = axes[1, 0]
channel_missed = df.groupby("channel")["replied"].apply(lambda x: (x == 0).mean() * 100).sort_values()
bars = ax.barh(channel_missed.index, channel_missed.values, color=channel_colors[::-1], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, channel_missed.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontweight="bold")
ax.set_title("Missed Rate by Channel", fontsize=12, fontweight="bold")
ax.set_xlabel("Missed Rate (%)")
ax.set_xlim(0, max(channel_missed.values) + 8)

# 4e. High Intent vs Missed Rate
ax = axes[1, 1]
intent_data = df.groupby("high_intent_flag")["replied"].apply(lambda x: (x == 0).mean() * 100)
bars = ax.bar(["Low Intent (0)", "High Intent (1)"], intent_data.values,
              color=["#95a5a6", "#e67e22"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, intent_data.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_title("Missed Rate by Intent Level", fontsize=12, fontweight="bold")
ax.set_ylabel("Missed Rate (%)")
ax.set_ylim(0, max(intent_data.values) + 5)

# 4f. Business Hours vs Missed Rate
ax = axes[1, 2]
hours_data = df.groupby("is_business_hours")["replied"].apply(lambda x: (x == 0).mean() * 100)
bars = ax.bar(["After Hours (0)", "Business Hours (1)"], hours_data.values,
              color=["#8e44ad", "#2ecc71"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, hours_data.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_title("Missed Rate: Business vs After Hours", fontsize=12, fontweight="bold")
ax.set_ylabel("Missed Rate (%)")
ax.set_ylim(0, max(hours_data.values) + 5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "eda_overview.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[EDA] Saved: eda_overview.png")

# ── 5. Correlation Heatmap ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[EDA] Saved: eda_correlation_heatmap.png")

# ── 6. Message Hour Distribution ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
hour_data = df.groupby(["message_hour", "replied"]).size().unstack(fill_value=0)
hour_data.columns = ["Missed", "Replied"]
hour_data.plot(kind="bar", ax=ax, color=["#e74c3c", "#27ae60"], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Leads")
ax.set_title("Lead Volume by Hour (Missed vs Replied)", fontsize=13, fontweight="bold")
ax.legend()
ax.set_xticks(range(0, len(hour_data), 1))
ax.set_xticklabels(hour_data.index, rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_hourly_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[EDA] Saved: eda_hourly_distribution.png")

# ── 7. Previous Contacts Analysis ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
prev_counts = df.groupby(["prev_contacts", "replied"]).size().unstack(fill_value=0)
prev_counts.columns = ["Missed", "Replied"]
prev_counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#27ae60"], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Previous Contacts")
ax.set_ylabel("Number of Leads")
ax.set_title("Missed Leads by Previous Contact Count", fontsize=13, fontweight="bold")
ax.legend()
ax.set_xticks(range(len(prev_counts)))
ax.set_xticklabels(prev_counts.index, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_prev_contacts.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[EDA] Saved: eda_prev_contacts.png")

# ── 8. Box Plot: Response Gap by Channel ───────────────────
fig, ax = plt.subplots(figsize=(10, 6))
channel_order = df.groupby("channel")["response_gap_hrs"].median().sort_values().index
sns.boxplot(data=df, x="channel", y="response_gap_hrs", order=channel_order,
            palette="Set2", ax=ax, showfliers=False)
ax.set_title("Response Gap Distribution by Channel", fontsize=13, fontweight="bold")
ax.set_xlabel("Channel")
ax.set_ylabel("Response Gap (hours)")
ax.set_ylim(0, 200)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_gap_by_channel.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[EDA] Saved: eda_gap_by_channel.png")

# ── 9. Intent Score Distribution ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
# Compute intent score
intent_words = ["price", "buy", "interested", "demo", "quote", "available"]
intent_scores = df["message_text"].apply(
    lambda t: sum(w in str(t).lower() for w in intent_words))
ax.hist(intent_scores[df["replied"] == 0], bins=range(0, 6), alpha=0.7,
        label="Missed", color="#e74c3c", edgecolor="white")
ax.hist(intent_scores[df["replied"] == 1], bins=range(0, 6), alpha=0.7,
        label="Replied", color="#27ae60", edgecolor="white")
ax.set_xlabel("Intent Score (number of intent keywords)")
ax.set_ylabel("Count")
ax.set_title("Intent Score Distribution: Missed vs Replied", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_intent_score.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[EDA] Saved: eda_intent_score.png")

# ── 10. Statistical Summary Report ──────────────────────────
print("\n" + "=" * 60)
print("  STATISTICAL SUMMARY REPORT")
print("=" * 60)

# Overall stats
total = len(df)
missed = (df["replied"] == 0).sum()
replied = (df["replied"] == 1).sum()
print(f"\nTotal Leads: {total}")
print(f"Missed Leads: {missed} ({missed/total*100:.1f}%)")
print(f"Replied Leads: {replied} ({replied/total*100:.1f}%)")

# Response gap stats
print(f"\nResponse Gap (hours):")
print(f"  Mean:     {df['response_gap_hrs'].mean():.1f}")
print(f"  Median:   {df['response_gap_hrs'].median():.1f}")
print(f"  Std Dev:  {df['response_gap_hrs'].std():.1f}")
print(f"  Min:      {df['response_gap_hrs'].min():.1f}")
print(f"  Max:      {df['response_gap_hrs'].max():.1f}")

# Channel breakdown
print(f"\nChannel Breakdown:")
for ch in df["channel"].unique():
    ch_df = df[df["channel"] == ch]
    ch_missed = (ch_df["replied"] == 0).sum()
    print(f"  {ch:15s}: {len(ch_df):3d} leads, {ch_missed:3d} missed ({ch_missed/len(ch_df)*100:.1f}%)")

# Hour analysis
print(f"\nBusiest Hours (top 5):")
hour_counts = df["message_hour"].value_counts().head(5).sort_index()
for h, c in hour_counts.items():
    print(f"  Hour {h:2d}: {c} leads")

print(f"\nQuietest Hours (bottom 5):")
hour_counts_tail = df["message_hour"].value_counts().tail(5).sort_index()
for h, c in hour_counts_tail.items():
    print(f"  Hour {h:2d}: {c} leads")

# Save as text report
report_lines = [
    "=" * 60,
    "  MISSED-LEAD DETECTOR - EDA REPORT",
    "=" * 60,
    f"\nGenerated from: {DATA}",
    f"Total Records: {total}",
    f"Features: {len(df.columns)}",
    "",
    f"Target Distribution:",
    f"  Missed: {missed} ({missed/total*100:.1f}%)",
    f"  Replied: {replied} ({replied/total*100:.1f}%)",
    "",
    f"Channels: {', '.join(df['channel'].unique())}",
    f"Hour Range: {df['message_hour'].min()}-{df['message_hour'].max()}",
    f"Avg Response Gap: {df['response_gap_hrs'].mean():.1f}h",
    f"High Intent Leads: {df['high_intent_flag'].sum()}",
    f"Avg Previous Contacts: {df['prev_contacts'].mean():.1f}",
]
with open(os.path.join(OUT_DIR, "eda_report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print(f"\n[EDA] All outputs saved to {OUT_DIR}")
print("[EDA] EDA complete!")
