"""
dashboard.py — Missed-Lead Detector
Interactive Streamlit dashboard showing REAL data from the live pipeline.
Displays results fetched from Gmail inbox → ML scoring → follow-up sending.

Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, json, pickle, subprocess, sys
from datetime import datetime
from PIL import Image

# ── Paths ──────────────────────────────────────────────────
BASE     = os.path.dirname(__file__)
SCORED   = os.path.join(BASE, "..", "outputs", "leads_scored.csv")
SEGMENT  = os.path.join(BASE, "..", "outputs", "leads_segmented.csv")
CM_IMG   = os.path.join(BASE, "..", "outputs", "confusion_matrix.png")
FI_IMG   = os.path.join(BASE, "..", "outputs", "feature_importance.png")
REPORT   = os.path.join(BASE, "..", "outputs", "classification_report.txt")
SENT_LOG = os.path.join(BASE, "..", "logs", "sent_leads.json")
EDA_DIR  = os.path.join(BASE, "..", "outputs", "eda")
MODEL_CMP = os.path.join(BASE, "..", "outputs", "model_comparison.json")
DL_HIST_IMG = os.path.join(BASE, "..", "outputs", "dl_training_history.png")
DL_CM_IMG   = os.path.join(BASE, "..", "outputs", "dl_confusion_matrix.png")
DL_ROC_IMG  = os.path.join(BASE, "..", "outputs", "dl_roc_curve.png")
CMP_CHART   = os.path.join(BASE, "..", "outputs", "model_comparison_chart.png")
XGB_TUNING  = os.path.join(BASE, "..", "outputs", "xgb_tuning_results.json")
MERGED_DATA = os.path.join(BASE, "..", "data", "leads_merged.csv")

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Missed-Lead Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; font-size: 1.05rem; }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .metric-value { font-size: 2.5rem; font-weight: 700; color: #1a1a2e; line-height: 1; }
    .metric-card .metric-label { font-size: 0.9rem; color: #555; margin-top: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .missed-card .metric-value { color: #e74c3c; }
    .replied-card .metric-value { color: #27ae60; }
    .auc-card .metric-value { color: #667eea; }
    .email-card .metric-value { color: #f39c12; }
    .inbox-card .metric-value { color: #3498db; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 { color: white !important; }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .status-live { background: #27ae60; color: white; }
    .status-demo { background: #f39c12; color: white; }
    .status-off { background: #95a5a6; color: white; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 Missed-Lead Detector</h1>
    <p>AI-Powered Missed-Lead Detection & Automated Follow-Up System — Live Backend</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["📊 Live Dashboard", "📋 Inbox Leads", "🤖 Model Results", "📧 Sent Follow-Ups", "📈 EDA", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 👥 Team")
    st.markdown("- **JAI VIJAI M** — 210425243091")
    st.markdown("- **ARUNKUMAR I** — 210425243028")
    st.markdown("### 🏫 Details")
    st.markdown("- CIT Chennai, AIDS")
    st.markdown("- Batch 2025-27")
    st.markdown("- SDG 8: Decent Work")

# ── Helper: Check if live mode is configured ───────────────
def is_live_configured() -> bool:
    return bool(os.getenv("IMAP_USER")) or bool(os.getenv("SMTP_USER"))

# ── Load Data ──────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data():
    scored = pd.read_csv(SCORED) if os.path.exists(SCORED) else pd.DataFrame()
    segmented = pd.read_csv(SEGMENT) if os.path.exists(SEGMENT) else pd.DataFrame()
    sent_ids = []
    if os.path.exists(SENT_LOG):
        with open(SENT_LOG) as f:
            sent_ids = json.load(f)
    return scored, segmented, sent_ids

scored_df, seg_df, sent_ids = load_data()

# ══════════════════════════════════════════════════════════
# PAGE: Live Dashboard
# ══════════════════════════════════════════════════════════
if page == "📊 Live Dashboard":
    st.markdown('<div class="section-header">📊 Live Pipeline Overview</div>', unsafe_allow_html=True)

    # Connection status
    live_configured = is_live_configured()
    status_text = "LIVE - Connected to Gmail" if live_configured else "DEMO - Using synthetic data"
    status_class = "status-live" if live_configured else "status-demo"
    st.markdown(f'Pipeline Status: <span class="status-badge {status_class}">{status_text}</span>', unsafe_allow_html=True)

    # Metrics from scored data (REAL inbox data)
    col1, col2, col3, col4 = st.columns(4)

    if len(scored_df) > 0:
        total_inbox = len(scored_df)
        high_intent = int(scored_df["high_intent_flag"].sum())
        avg_gap = scored_df["response_gap_hrs"].mean()
        predicted_missed = int(scored_df["predicted_missed"].sum())

        with col1:
            st.markdown(f"""
            <div class="metric-card inbox-card">
                <div class="metric-value">{total_inbox}</div>
                <div class="metric-label">Inbox Emails Fetched</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card auc-card">
                <div class="metric-value">{high_intent}</div>
                <div class="metric-label">High-Intent Inquiries</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card missed-card">
                <div class="metric-value">{predicted_missed}</div>
                <div class="metric-label">ML-Detected Missed</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card email-card">
                <div class="metric-value">{len(sent_ids)}</div>
                <div class="metric-label">Follow-Ups Sent</div>
            </div>""", unsafe_allow_html=True)
    else:
        with col1:
            st.markdown(f"""
            <div class="metric-card inbox-card">
                <div class="metric-value">0</div>
                <div class="metric-label">Inbox Emails Fetched</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card auc-card">
                <div class="metric-value">0</div>
                <div class="metric-label">High-Intent Inquiries</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card missed-card">
                <div class="metric-value">0</div>
                <div class="metric-label">ML-Detected Missed</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card email-card">
                <div class="metric-value">{len(sent_ids)}</div>
                <div class="metric-label">Follow-Ups Sent</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Recent inbox emails table
    if len(scored_df) > 0:
        st.markdown('<div class="section-header">📥 Recent Customer Emails from Inbox</div>', unsafe_allow_html=True)

        display_cols = ["lead_id", "_customer_name", "_customer_email", "_subject",
                        "response_gap_hrs", "high_intent_flag", "missed_probability", "predicted_missed"]
        display_df = scored_df[display_cols].copy()
        display_df.columns = ["Lead ID", "Customer", "Email", "Subject",
                              "Gap (hrs)", "High Intent", "Missed Prob.", "Predicted Missed"]
        display_df["Missed Prob."] = display_df["Missed Prob."].apply(lambda x: f"{x:.1%}")
        display_df["High Intent"] = display_df["High Intent"].map({1: "✅ YES", 0: "❌ NO"})
        display_df["Predicted Missed"] = display_df["Predicted Missed"].map({1: "🔴 MISSED", 0: "🟢 OK"})

        st.dataframe(display_df, use_container_width=True, height=300)
        st.caption(f"Showing {len(display_df)} inbox emails scored by the ML model")
    else:
        st.info("📭 No inbox emails fetched yet. Run `python src/orchestrator.py --live` to fetch from Gmail.")

    # Gap distribution chart
    if len(scored_df) > 0:
        st.markdown('<div class="section-header">⏱️ Response Gaps (Real Inbox Data)</div>', unsafe_allow_html=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.5))
        gaps = scored_df["response_gap_hrs"]
        ax.hist(gaps, bins=20, alpha=0.7, color="#667eea", edgecolor="white")
        ax.axvline(gaps.mean(), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {gaps.mean():.1f}h")
        ax.set_xlabel("Response Gap (Hours)")
        ax.set_ylabel("Count")
        ax.set_title("Response Gap Distribution — Scored Inbox Emails")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Run pipeline button
    st.markdown("---")
    st.markdown("#### 🔄 Run Pipeline Now")
    st.markdown("Fetch new emails from Gmail, score them, and send follow-ups.")

    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        if st.button("🚀 Run Preview (no send)", use_container_width=True):
            with st.spinner("Fetching emails from Gmail and scoring..."):
                try:
                    env = os.environ.copy()
                    result = subprocess.run(
                        [sys.executable, "src/orchestrator.py", "--preview"],
                        cwd=os.path.dirname(BASE),
                        capture_output=True, text=True, timeout=120, env=env
                    )
                    st.text(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                    if result.stderr:
                        st.text(result.stderr[-1000:])
                    st.cache_data.clear()
                    st.rerun()
                except subprocess.TimeoutExpired:
                    st.error("Pipeline timed out. Check your Gmail connection.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with run_col2:
        if st.button("⚡ Run LIVE (send follow-ups)", use_container_width=True, type="primary"):
            if not live_configured:
                st.warning("⚠️ SMTP not configured. Set SMTP_USER and SMTP_PASS env vars.")
            else:
                with st.spinner("Fetching, scoring, and sending follow-ups..."):
                    try:
                        env = os.environ.copy()
                        result = subprocess.run(
                            [sys.executable, "src/orchestrator.py", "--live"],
                            cwd=os.path.dirname(BASE),
                            capture_output=True, text=True, timeout=120, env=env
                        )
                        st.text(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                        if result.stderr:
                            st.text(result.stderr[-1000:])
                        st.cache_data.clear()
                        st.rerun()
                    except subprocess.TimeoutExpired:
                        st.error("Pipeline timed out.")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE: Inbox Leads
# ══════════════════════════════════════════════════════════
elif page == "📋 Inbox Leads":
    st.markdown('<div class="section-header">📋 Scored Inbox Leads (Live Data)</div>', unsafe_allow_html=True)

    if len(scored_df) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_status = st.selectbox("Filter by Status", ["All", "Missed Only", "High Intent Only"])
        with col2:
            customers = sorted(scored_df["_customer_name"].unique().tolist())
            filter_customer = st.selectbox("Filter by Customer", ["All"] + customers)
        with col3:
            sort_by = st.selectbox("Sort by", ["missed_probability", "response_gap_hrs", "lead_id"])

        display_df = scored_df.copy()
        if filter_status == "Missed Only":
            display_df = display_df[display_df["predicted_missed"] == 1]
        elif filter_status == "High Intent Only":
            display_df = display_df[display_df["high_intent_flag"] == 1]
        if filter_customer != "All":
            display_df = display_df[display_df["_customer_name"] == filter_customer]

        display_df = display_df.sort_values(sort_by, ascending=False)

        # Show full details including body preview
        show_cols = ["lead_id", "_customer_name", "_customer_email", "_subject",
                     "message_text", "response_gap_hrs", "high_intent_flag",
                     "missed_probability", "predicted_missed", "_received_time"]
        table_df = display_df[show_cols].copy()
        table_df.columns = ["Lead ID", "Customer", "Email", "Subject",
                            "Message", "Gap (hrs)", "Intent", "Prob.", "Missed?", "Received"]
        table_df["Prob."] = table_df["Prob."].apply(lambda x: f"{x:.2%}")
        table_df["Intent"] = table_df["Intent"].map({1: "HIGH", 0: "low"})
        table_df["Missed?"] = table_df["Missed?"].map({1: "🔴 YES", 0: "🟢 No"})
        table_df["Message"] = table_df["Message"].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else x)

        st.dataframe(table_df, use_container_width=True, height=500)
        st.caption(f"Showing {len(table_df)} of {len(scored_df)} inbox leads")
    else:
        st.warning("📭 No inbox data. Run the pipeline first.")

# ══════════════════════════════════════════════════════════
# PAGE: Model Results
# ══════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    # Detect data source
    data_source = "Merged (Kaggle + Synthetic, 13,740 rows)" if os.path.exists(MERGED_DATA) else "Synthetic (500 rows)"
    st.markdown(f'<div class="section-header">🤖 Model Performance — {data_source}</div>', unsafe_allow_html=True)

    # Load actual metrics from classification report
    ensemble_acc = 0.82
    ensemble_f1_missed = 0.59
    ensemble_precision_missed = 0.68
    ensemble_recall_missed = 0.52
    if os.path.exists(REPORT):
        with open(REPORT) as f:
            report_text = f.read()
        for line in report_text.split('\n'):
            parts = line.split()
            if len(parts) >= 5 and parts[0] == 'accuracy':
                ensemble_acc = float(parts[1])
            if len(parts) >= 5 and parts[0] == 'Missed':
                ensemble_precision_missed = float(parts[1])
                ensemble_recall_missed = float(parts[2])
                ensemble_f1_missed = float(parts[3])

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card auc-card">
            <div class="metric-value">{ensemble_acc:.0%}</div>
            <div class="metric-label">Overall Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card auc-card">
            <div class="metric-value">{ensemble_f1_missed:.3f}</div>
            <div class="metric-label">Missed F1 Score</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card auc-card">
            <div class="metric-value">{ensemble_precision_missed:.3f}</div>
            <div class="metric-label">Missed Precision</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card auc-card">
            <div class="metric-value">{ensemble_recall_missed:.3f}</div>
            <div class="metric-label">Missed Recall</div>
        </div>""", unsafe_allow_html=True)

    # Load model comparison from JSON if available
    model_comparison = {}
    if os.path.exists(MODEL_CMP):
        with open(MODEL_CMP) as f:
            model_comparison = json.load(f)

    # Model comparison table — from real training results
    st.markdown("#### Model Comparison (Test AUC)")
    if model_comparison.get("models"):
        cmp_rows = []
        for name, info in sorted(model_comparison["models"].items(),
                                 key=lambda x: x[1]["auc"], reverse=True):
            badge = " 🏆" if name == model_comparison.get("best", "") else ""
            cmp_rows.append({"Model": name + badge, "Test AUC": info["auc"]})
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python src/train_model.py` to generate model comparison data.")

    # Charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Grand Ensemble Confusion Matrix")
        if os.path.exists(CM_IMG):
            st.image(Image.open(CM_IMG), use_container_width=True)
        else:
            st.warning("Run `python src/train_model.py` to generate.")

    with col2:
        st.markdown("#### Feature Importance (Random Forest)")
        if os.path.exists(FI_IMG):
            st.image(Image.open(FI_IMG), use_container_width=True)
        else:
            st.warning("Run `python src/train_model.py` to generate.")

    # Classification Report
    st.markdown("#### Grand Ensemble Classification Report")
    if os.path.exists(REPORT):
        with open(REPORT) as f:
            st.code(f.read(), language="text")
    else:
        st.warning("Run `python src/train_model.py` to generate.")

    # Before/After Tuning Chart
    st.markdown("---")
    st.markdown("<div class=\"section-header\">📊 Before vs After Optuna Tuning</div>", unsafe_allow_html=True)
    if os.path.exists(CMP_CHART):
        st.image(Image.open(CMP_CHART), use_container_width=True)
    else:
        st.info("Run `python src/plot_model_comparison.py` to generate the tuning comparison chart.")

    # XGBoost Tuning Results
    if os.path.exists(XGB_TUNING):
        with open(XGB_TUNING) as f:
            tuning = json.load(f)
        st.markdown("#### XGBoost Optuna Tuning Results")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(f"""
            <div class="metric-card auc-card">
                <div class="metric-value">{tuning.get('test_auc', 0):.4f}</div>
                <div class="metric-label">Tuned Test AUC</div>
            </div>""", unsafe_allow_html=True)
        with tc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{tuning.get('n_trials', 0)}</div>
                <div class="metric-label">Optuna Trials</div>
            </div>""", unsafe_allow_html=True)
        with tc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{tuning.get('n_samples', 0):,}</div>
                <div class="metric-label">Training Samples</div>
            </div>""", unsafe_allow_html=True)
        if tuning.get("best_params"):
            st.json(tuning["best_params"])

    # Deep Learning section
    st.markdown("---")
    st.markdown("<div class=\"section-header\">🧠 Deep Learning Model (PyTorch GPU)</div>",
                unsafe_allow_html=True)

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    dl_auc = model_comparison.get("models", {}).get("DeepLearning(PyTorch)", {}).get("auc", 0.896)
    with dl_col1:
        st.markdown(f"""
        <div class="metric-card auc-card">
            <div class="metric-value">{dl_auc:.3f}</div>
            <div class="metric-label">DL Test AUC</div>
        </div>""", unsafe_allow_html=True)
    with dl_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">RTX 3050</div>
            <div class="metric-label">GPU Accelerated</div>
        </div>""", unsafe_allow_html=True)
    with dl_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">ResNet</div>
            <div class="metric-label">Architecture</div>
        </div>""", unsafe_allow_html=True)

    dl_charts = st.columns(3)
    with dl_charts[0]:
        if os.path.exists(DL_HIST_IMG):
            st.image(Image.open(DL_HIST_IMG), caption="Training History", use_container_width=True)
    with dl_charts[1]:
        if os.path.exists(DL_CM_IMG):
            st.image(Image.open(DL_CM_IMG), caption="DL Confusion Matrix", use_container_width=True)
    with dl_charts[2]:
        if os.path.exists(DL_ROC_IMG):
            st.image(Image.open(DL_ROC_IMG), caption="DL ROC Curve", use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE: Sent Follow-Ups
# ══════════════════════════════════════════════════════════
elif page == "📧 Sent Follow-Ups":
    st.markdown('<div class="section-header">📧 Follow-Up Email Log (Live Data)</div>', unsafe_allow_html=True)

    # Find which leads from scored_df have been sent follow-ups
    sent_lead_ids = set(sent_ids)

    total_sent = len(sent_lead_ids)
    st.markdown(f"""
    <div class="metric-card email-card" style="max-width:300px; margin-bottom:1.5rem;">
        <div class="metric-value">{total_sent}</div>
        <div class="metric-label">Follow-Up Emails Sent</div>
    </div>""", unsafe_allow_html=True)

    if total_sent > 0:
        # Check which scored leads were sent follow-ups
        if len(scored_df) > 0:
            scored_sent = scored_df[scored_df["lead_id"].isin(sent_lead_ids)]
            if len(scored_sent) > 0:
                st.markdown("#### Recently Sent Follow-Ups")
                sent_display = scored_sent[["lead_id", "_customer_name", "_customer_email", "_subject", "missed_probability"]].copy()
                sent_display.columns = ["Lead ID", "Customer", "Email", "Subject", "Missed Prob."]
                sent_display["Missed Prob."] = sent_display["Missed Prob."].apply(lambda x: f"{x:.1%}")
                st.dataframe(sent_display, use_container_width=True)

        st.markdown("#### All Emailed Lead IDs")
        all_ids = sorted(sent_lead_ids)
        cols_per_row = 8
        for i in range(0, len(all_ids), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(all_ids):
                    col.code(all_ids[idx])

        st.info("💡 These leads have been emailed and won't receive duplicates on re-run.")
    else:
        st.warning("📭 No follow-up emails sent yet. Run the pipeline with `--live`.")

# ══════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════
elif page == "📈 EDA":
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    eda_count = 500 if not os.path.exists(MERGED_DATA) else 13740
    eda_source = "Synthetic" if not os.path.exists(MERGED_DATA) else "Merged (Kaggle + Synthetic)"
    st.markdown(f"EDA visualizations generated from the {eda_source} dataset ({eda_count:,} leads).")

    eda_images = {
        "Overview": "eda_overview.png",
        "Correlation Heatmap": "eda_correlation_heatmap.png",
        "Hourly Distribution": "eda_hourly_distribution.png",
        "Previous Contacts": "eda_prev_contacts.png",
        "Gap by Channel": "eda_gap_by_channel.png",
        "Intent Score": "eda_intent_score.png",
    }

    selected_chart = st.selectbox("Select Visualization", list(eda_images.keys()))
    img_path = os.path.join(EDA_DIR, eda_images[selected_chart])
    if os.path.exists(img_path):
        st.image(Image.open(img_path), use_container_width=True)
    else:
        st.warning("Run `python notebooks/EDA.py` to generate this visualization.")

    # Quick stats
    st.markdown("#### Dataset Summary")
    if os.path.exists(MERGED_DATA):
        st.markdown("""
        - **Total Leads:** 13,740 (500 synthetic + 9,240 X Education + 4,000 Customer Support)
        - **Channels:** Email, Phone Inquiry, Website Chat, WhatsApp
        - **Best Model:** XGBoost (Optuna-tuned, AUC 0.9794)
        - **Models:** LR, NB, DT, RF, XGBoost + Ensemble + Deep Learning (PyTorch GPU) + Grand Ensemble
        - **Features:** 9 engineered features (channel, intent, hour, gap, etc.)
        - **Tuning:** 100 Optuna trials, 5-fold CV
        """)
    else:
        st.markdown("""
        - **Total Leads:** 500 synthetic CRM records
        - **Channels:** WhatsApp, Email, Website Chat, Phone Inquiry
        - **Missed Rate:** ~25.4%
        - **Features:** 9 engineered features (channel, intent, hour, gap, etc.)
        - **Models:** LR, NB, DT, RF, XGBoost + Ensemble + K-Means
        """)

# ══════════════════════════════════════════════════════════
# PAGE: About
# ══════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Machine Learning-Based Missed-Lead Detection and Automated Follow-Up System

    **Objective:** To design a machine learning system that automatically identifies "missed leads" —
    customers who expressed interest but did not receive a timely response — and triggers either a
    staff alert or an automated follow-up email.

    ---

    #### 🏗️ System Architecture
    ```
    GMAIL INBOX (IMAP) → FILTER NEWSLETTERS → FEATURE ENGINEERING
            ↓
    ML PIPELINE (LR, NB, DT, RF, XGBoost, Ensemble, K-Means)
            ↓
    INFERENCE: missed_probability ≥ 0.50 → MISSED LEAD
            ↓                        ↓
    auto_followup.py      employee_reminder.py
    (SMTP threaded email)  (repeating popup)
    ```

    #### 🔄 Live Pipeline
    ```
    python src/orchestrator.py --preview   ← Preview without sending
    python src/orchestrator.py --live      ← Fetch & send follow-ups
    ```

    #### 🎯 SDG Alignment
    **SDG 8 — Decent Work and Economic Growth**

    ---

    #### 📦 Tech Stack
    - **Python 3.11** — Core language
    - **Scikit-learn** — ML models & evaluation
    - **XGBoost** — Gradient boosting classifier (Optuna-tuned)
    - **PyTorch** — Deep learning model (GPU-accelerated on RTX 3050)
    - **Optuna** — Hyperparameter optimization (100 trials)
    - **Kaggle** — Real-world training data (X Education + Customer Support)
    - **Pandas / NumPy** — Data processing
    - **Matplotlib / Seaborn** — Visualization
    - **Streamlit** — Dashboard UI
    - **IMAP** — Gmail inbox fetching
    - **SMTP** — Email sending
    - **Tkinter** — Desktop popup alerts
    - **Windows Task Scheduler** — Daily automation
    """)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.85rem;'>"
    "Missed-Lead Detector · JAI VIJAI M & ARUNKUMAR I · CIT Chennai · Batch 2025-27 · "
    f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    "</p>",
    unsafe_allow_html=True
)
