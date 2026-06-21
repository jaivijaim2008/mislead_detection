"""
dashboard.py — Missed-Lead Detector
Aurora Borealis Dashboard — Premium Modern Analytics Command Center.

Covers live monitoring, lead browsing, intent preview, ML visualisations,
and business parameters editor.

Design: Aurora Borealis dark theme with smooth gradients, glassmorphism cards,
and refined micro-interactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import subprocess
import sys
from datetime import datetime

# ── Import Fallbacks & Safety ──────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Paths ──────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
SCORED    = os.path.join(BASE, "..", "outputs", "leads_scored.csv")
CM_IMG    = os.path.join(BASE, "..", "outputs", "confusion_matrix.png")
FI_IMG    = os.path.join(BASE, "..", "outputs", "feature_importance.png")
REPORT    = os.path.join(BASE, "..", "outputs", "classification_report.txt")
SENT_LOG  = os.path.join(BASE, "..", "logs", "sent_leads.json")
REPLY_LOG = os.path.join(BASE, "..", "logs", "auto_replies.json")
NOTIF_LOG = os.path.join(BASE, "..", "logs", "notifications.json")
FOLLOWUP_LOG = os.path.join(BASE, "..", "logs", "followup_status.json")
MODEL_CMP = os.path.join(BASE, "..", "outputs", "model_comparison.json")
DL_HIST   = os.path.join(BASE, "..", "outputs", "dl_training_history.png")
DL_CM     = os.path.join(BASE, "..", "outputs", "dl_confusion_matrix.png")
DL_ROC    = os.path.join(BASE, "..", "outputs", "dl_roc_curve.png")
CMP_CHART = os.path.join(BASE, "..", "outputs", "model_comparison_chart.png")
XGB_TUNING = os.path.join(BASE, "..", "outputs", "xgb_tuning_results.json")
OVERRIDES_PATH = os.path.join(BASE, "..", "logs", "config_overrides.json")

# Import config constants directly for default previewing
sys.path.insert(0, BASE)
import config

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Missed-Lead Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Design System (Aurora Borealis Dark) ──────────────
DESIGN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #0a0a12 0%, #0d1117 30%, #0a0f1a 60%, #0a0a12 100%) !important;
        color: #e6edf3 !important;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(88, 166, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(88, 166, 255, 0.08) !important;
    }

    /* ── Aurora Glow Header ──────────────────────────────── */
    .aurora-header {
        position: relative;
        background: linear-gradient(135deg, rgba(13, 17, 23, 0.8) 0%, rgba(16, 20, 28, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(88, 166, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 2rem;
        overflow: hidden;
    }
    .aurora-header::before {
        content: '';
        position: absolute;
        top: -80px; right: -60px;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.12) 0%, rgba(139, 92, 246, 0.06) 40%, transparent 70%);
        animation: aurora-pulse 8s ease-in-out infinite;
        pointer-events: none;
    }
    .aurora-header::after {
        content: '';
        position: absolute;
        bottom: -40px; left: -30px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.08) 0%, transparent 70%);
        animation: aurora-pulse 6s ease-in-out infinite reverse;
        pointer-events: none;
    }
    @keyframes aurora-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.1); }
    }
    .aurora-header h1 {
        font-weight: 900;
        font-size: 2.4rem;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #e6edf3 20%, #38bdf8 50%, #8b5cf6 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .aurora-header p {
        color: #7d8590;
        margin-top: 0.5rem;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }

    /* ── Glass Cards ─────────────────────────────────────── */
    .glass-card {
        background: rgba(13, 17, 23, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(88, 166, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(88, 166, 255, 0.12);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }

    /* ── KPI Metric Cards ────────────────────────────────── */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.75rem;
    }
    .kpi-card {
        flex: 1;
        background: rgba(13, 17, 23, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.06);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.3);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .kpi-card.blue::before  { background: linear-gradient(90deg, #38bdf8, #0ea5e9); }
    .kpi-card.red::before   { background: linear-gradient(90deg, #f87171, #ef4444); }
    .kpi-card.green::before { background: linear-gradient(90deg, #34d399, #10b981); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
    .kpi-card.purple::before{ background: linear-gradient(90deg, #a78bfa, #8b5cf6); }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #7d8590;
        margin-top: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .kpi-card.blue .kpi-value   { color: #38bdf8; }
    .kpi-card.red .kpi-value    { color: #f87171; }
    .kpi-card.green .kpi-value  { color: #34d399; }
    .kpi-card.amber .kpi-value  { color: #fbbf24; }
    .kpi-card.purple .kpi-value { color: #a78bfa; }

    /* ── Status Badges ───────────────────────────────────── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-live {
        background: rgba(52, 211, 153, 0.1);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .badge-live::before {
        content: '';
        width: 6px; height: 6px;
        background: #34d399;
        border-radius: 50%;
        animation: pulse-dot 2s infinite;
    }
    .badge-demo {
        background: rgba(251, 191, 36, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* ── Notification Cards ──────────────────────────────── */
    .notif-card {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(88, 166, 255, 0.06);
        border-left: 3px solid #38bdf8;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        transition: background 0.2s ease;
    }
    .notif-card:hover { background: rgba(30, 38, 48, 0.5); }
    .notif-card.new_lead   { border-left-color: #38bdf8; }
    .notif-card.auto_reply { border-left-color: #34d399; }
    .notif-card.overdue    { border-left-color: #f87171; }
    .notif-card.info       { border-left-color: #a78bfa; }
    .notif-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #e6edf3;
    }
    .notif-time {
        font-size: 0.72rem;
        color: #484f58;
        font-family: 'JetBrains Mono', monospace;
    }
    .notif-msg {
        font-size: 0.82rem;
        color: #7d8590;
        line-height: 1.4;
        margin-top: 0.2rem;
    }

    /* ── Terminal Output ─────────────────────────────────── */
    .terminal-box {
        background: #0d1117 !important;
        border: 1px solid rgba(88, 166, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        color: #7ee787 !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4) !important;
        line-height: 1.5 !important;
        max-height: 400px;
        overflow-y: auto;
    }

    /* ── Email Mock ──────────────────────────────────────── */
    .email-mock {
        background: rgba(13, 17, 23, 0.6);
        border: 1px solid rgba(88, 166, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    .email-mock-header {
        border-bottom: 1px solid rgba(88, 166, 255, 0.06);
        padding-bottom: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.82rem;
        color: #7d8590;
    }
    .email-mock-header span { color: #e6edf3; font-weight: 500; }
    .email-mock-body {
        font-size: 0.88rem;
        color: #e6edf3;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    /* ── Tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: rgba(13, 17, 23, 0.6);
        border-radius: 10px;
        padding: 0.3rem;
        border: 1px solid rgba(88, 166, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #7d8590 !important;
        font-weight: 600;
        padding: 0.4rem 1.25rem;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
    }

    /* ── Input Overrides ─────────────────────────────────── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    textarea {
        background: rgba(22, 27, 34, 0.6) !important;
        border: 1px solid rgba(88, 166, 255, 0.1) !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }

    /* ── Button Overrides ────────────────────────────────── */
    .stButton > button {
        background: rgba(22, 27, 34, 0.6) !important;
        color: #e6edf3 !important;
        border: 1px solid rgba(88, 166, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.1) !important;
        transform: translateY(-1px);
    }
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.25) !important;
    }
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.35) !important;
    }

    /* ── Sidebar Status Box ──────────────────────────────── */
    .sidebar-status {
        background: rgba(22, 27, 34, 0.5);
        border: 1px solid rgba(88, 166, 255, 0.06);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-size: 0.82rem;
        line-height: 1.7;
    }

    /* ── Hide Streamlit Elements ─────────────────────────── */
    #MainMenu, footer { visibility: hidden; }

    /* ── Footer ──────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        color: #484f58;
        font-size: 0.78rem;
        border-top: 1px solid rgba(88, 166, 255, 0.04);
        margin-top: 4rem;
        letter-spacing: 0.04em;
    }
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# ── Dynamic Configuration Overrides Handler ────────────────
def load_overrides():
    if os.path.exists(OVERRIDES_PATH):
        try:
            with open(OVERRIDES_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_overrides(data):
    os.makedirs(os.path.dirname(OVERRIDES_PATH), exist_ok=True)
    with open(OVERRIDES_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)
    st.cache_data.clear()

overrides = load_overrides()

# ── Data Loading Helpers ──────────────────────────────────
@st.cache_data(ttl=15)
def load_scored_leads():
    if os.path.exists(SCORED):
        df = pd.read_csv(SCORED)
        if "replied" in df.columns and "predicted_missed" not in df.columns:
            # Map replied to predicted_missed
            # replied=1 means lead responded (not missed), replied=0 means missed
            df["predicted_missed"] = df["replied"].map({1: 0, 0: 1})
        return df
    return pd.DataFrame()

@st.cache_data(ttl=15)
def load_json_log(path):
    """Load a JSON log file. Returns list or dict based on actual content."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return data
        except Exception:
            pass
    # Default based on filename convention
    return {} if "followup" in path.lower() or "config" in path.lower() else []

# ── Connection Checks ──────────────────────────────────────
def is_live_mode() -> bool:
    return bool(os.getenv("IMAP_USER")) or bool(os.getenv("SMTP_USER"))

# ── Header Section ─────────────────────────────────────────
live_connected = is_live_mode()
status_label = "Gmail Connected — Live Data" if live_connected else "Gmail Not Connected"
status_class = "badge-live" if live_connected else "badge-demo"

st.markdown(f"""
<div class="aurora-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1>Missed-Lead Command Center</h1>
        <span class="badge {status_class}">{status_label}</span>
    </div>
    <p>AI-Powered Email Monitoring &bull; Smart Auto-Replies &bull; Sales Pipeline Retention Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configurations & Navigation ───────────────────
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to:",
        ["Command Center", "Lead Explorer", "Auto-Replies Tracker",
         "Interactive Pipeline Graph", "Model Analytics", "Workflow Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### System Health")

    leads_df = load_scored_leads()
    notifs = load_json_log(NOTIF_LOG)
    unread_notifs = [n for n in notifs if not n.get("read", False)] if isinstance(notifs, list) else []

    inbox_icon = "🟢" if live_connected else "🔴"
    st.markdown(f"""
    <div class="sidebar-status">
        <div><b>Gmail Connection</b>: {inbox_icon} {'Active' if live_connected else 'Not Connected'}</div>
        <div><b>Total Scored</b>: {len(leads_df)} leads</div>
        <div><b>Unread Alerts</b>: <span style="color: {'#f87171' if len(unread_notifs) > 0 else '#7d8590'}; font-weight: bold;">{len(unread_notifs)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size: 0.72rem; color:#484f58; text-align: center;'>Missed-Lead Detector v2.0<br>Batch 2025-27</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Command Center
# ══════════════════════════════════════════════════════════
if page == "Command Center":
    scored = load_scored_leads()
    reply_log = load_json_log(REPLY_LOG)
    followup_status = load_json_log(FOLLOWUP_LOG)

    total_leads = len(scored)
    missed_leads = int(scored["predicted_missed"].sum()) if "predicted_missed" in scored.columns and total_leads > 0 else 0
    replied_count = len(reply_log) if isinstance(reply_log, list) else 0

    overdue_count = 0
    if isinstance(followup_status, dict):
        overdue_count = sum(1 for s in followup_status.values()
                            if isinstance(s, dict) and s.get("auto_replied") and not s.get("human_followed_up"))

    high_intent = int(scored["high_intent_flag"].sum()) if "high_intent_flag" in scored.columns and total_leads > 0 else 0

    # ── Empty state when no data exists ───────────────────
    if total_leads == 0:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding: 3rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📬</div>
            <h2 style="color: #e6edf3; margin-bottom: 0.5rem;">No Inbox Data Yet</h2>
            <p style="color: #7d8590; font-size: 1rem; max-width: 520px; margin: 0 auto;">
                Connect your Gmail account to start scanning real customer emails.<br>
                The system will detect missed leads, score them with ML, and auto-reply.
            </p>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(22,27,34,0.5); border-radius: 10px; border: 1px solid rgba(88,166,255,0.08); max-width: 550px; margin-left: auto; margin-right: auto; text-align: left;">
                <p style="color: #fbbf24; font-weight: 600; margin-bottom: 0.5rem;">⚡ Setup Steps:</p>
                <p style="color: #7d8590; font-size: 0.85rem; line-height: 1.8;">
                    1. Enable 2FA on your Gmail and generate an App Password<br>
                    2. Add these to Streamlit Cloud secrets:<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:#38bdf8;">IMAP_USER</code> = your Gmail address<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:#38bdf8;">IMAP_PASS</code> = your App Password<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:#38bdf8;">SMTP_USER</code> = your Gmail address<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:#38bdf8;">SMTP_PASS</code> = your App Password<br>
                    3. Click "Trigger Scan Now" below to fetch real emails
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card blue">
            <div class="kpi-value">{total_leads:,}</div>
            <div class="kpi-label">Scanned Leads</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-value">{missed_leads}</div>
            <div class="kpi-label">Missed Leads</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-value">{replied_count}</div>
            <div class="kpi-label">Auto-Replied</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-value">{overdue_count}</div>
            <div class="kpi-label">Awaiting Action</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-value">{high_intent}</div>
            <div class="kpi-label">High Intent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Layout ────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Pipeline Controls
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📧 Inbox Monitoring Pipeline")

        c1, c2 = st.columns(2)
        with c1:
            scan_btn = st.button("Trigger Scan Now", type="primary", use_container_width=True)
        with c2:
            dry_btn = st.button("Simulate Dry-Run Scan", use_container_width=True)

        if scan_btn or dry_btn:
            is_dry = bool(dry_btn)
            st.markdown("**Pipeline Output:**")
            with st.spinner("Processing mailbox via SMTP/IMAP..."):
                try:
                    cmd = [sys.executable, os.path.join(BASE, "inbox_monitor.py")]
                    if is_dry:
                        cmd.append("--dry-run")
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    stdout = res.stdout if res.stdout else "No output returned."
                    st.markdown(f"<pre class='terminal-box'>{stdout}</pre>", unsafe_allow_html=True)
                    if res.stderr:
                        st.warning(f"Warnings:\n{res.stderr}")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution Error: {e}")
        elif not live_connected:
            st.warning("⚠️ Gmail not connected. Set IMAP_USER and IMAP_PASS in Streamlit secrets to enable live scanning.")
        else:
            st.markdown("<p style='color: #7d8590; font-style: italic; margin-top:0.5rem;'>Trigger a pipeline scan to read Gmail inboxes, run ML predictions, and execute automatic replies.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Distribution Chart
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📊 Lead Risk Distribution")

        if total_leads > 0 and "missed_probability" in scored.columns:
            if HAS_PLOTLY:
                fig = px.histogram(
                    scored, x="missed_probability", nbins=20,
                    title="Lead Risk Distribution (Threshold = 0.50)",
                    labels={"missed_probability": "Predicted Missed Probability", "count": "Lead Count"},
                    color_discrete_sequence=["#f87171"]
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#7d8590',
                    title_font_color='#e6edf3',
                    showlegend=False,
                    xaxis=dict(showgrid=False, linecolor='rgba(88,166,255,0.06)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(88,166,255,0.06)', linecolor='rgba(88,166,255,0.06)'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="#fbbf24",
                              annotation_text="Score Threshold", annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)
            else:
                if os.path.exists(CM_IMG) and HAS_PIL:
                    st.image(Image.open(CM_IMG), use_container_width=True)
        else:
            st.info("No lead metrics scored. Trigger a scan above to ingest data.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("🔔 Notification Feed")

        if isinstance(notifs, list) and notifs:
            if len(unread_notifs) > 0:
                if st.button(f"Mark all read ({len(unread_notifs)})", use_container_width=True):
                    for n in notifs:
                        n["read"] = True
                    with open(NOTIF_LOG, "w") as f:
                        json.dump(notifs, f, indent=2, default=str)
                    st.rerun()

            unread_count = 0
            for i, n in enumerate(reversed(notifs)):
                if not n.get("read", False) and unread_count < 10:
                    ntype = n.get("type", "info")
                    icon = {"new_lead": "📧", "auto_reply": "🤖", "overdue": "⚠️"}.get(ntype, "ℹ️")

                    st.markdown(f"""
                    <div class="notif-card {ntype}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                            <span class="notif-title">{icon} {n.get('title', 'Notification')}</span>
                            <span class="notif-time">{n.get('timestamp', '')[11:16]}</span>
                        </div>
                        <div class="notif-msg">{n.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    unread_count += 1
            if unread_count == 0:
                st.markdown("<p style='text-align:center; color:#484f58; padding: 2rem 0;'>All caught up! No unread notifications.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center; color:#484f58; padding: 2rem 0;'>No notifications found.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Lead Explorer
# ══════════════════════════════════════════════════════════
elif page == "Lead Explorer":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔍 Interactive Lead Table & Inspector")

    scored = load_scored_leads()

    if scored.empty:
        st.warning("No leads recorded. Please trigger an inbox scan to fetch records.")
    else:
        has_gmail_headers = "_customer_name" in scored.columns

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            q_search = st.text_input("Search Customer / Email", "")
        with c2:
            q_status = st.selectbox("Pipeline Status", ["All", "Missed Leads", "Responded (Low risk)"])
        with c3:
            q_intent = st.selectbox("High Intent Only", ["All", "Yes", "No"])
        with c4:
            channels = list(scored["channel"].unique()) if "channel" in scored.columns else ["Gmail"]
            q_channel = st.selectbox("Channel Source", ["All"] + channels)

        filtered = scored.copy()
        if q_search:
            s_pat = q_search.lower()
            if has_gmail_headers:
                filtered = filtered[
                    filtered["_customer_name"].str.lower().str.contains(s_pat, na=False) |
                    filtered["_customer_email"].str.lower().str.contains(s_pat, na=False)
                ]
            else:
                filtered = filtered[filtered["message_text"].str.lower().str.contains(s_pat, na=False)]

        if q_status != "All":
            val = 1 if q_status == "Missed Leads" else 0
            filtered = filtered[filtered["predicted_missed"] == val]

        if q_intent != "All":
            val = 1 if q_intent == "Yes" else 0
            filtered = filtered[filtered["high_intent_flag"] == val]

        if q_channel != "All" and "channel" in filtered.columns:
            filtered = filtered[filtered["channel"] == q_channel]

        # Display columns
        if has_gmail_headers:
            show_cols = ["lead_id", "_customer_name", "_customer_email", "_subject", "channel",
                         "response_gap_hrs", "high_intent_flag", "missed_probability", "predicted_missed"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            rename_map = {
                "lead_id": "Lead ID", "_customer_name": "Customer",
                "_customer_email": "Email", "_subject": "Subject",
                "channel": "Source", "response_gap_hrs": "Gap (Hrs)",
                "high_intent_flag": "High Intent", "missed_probability": "Risk Score",
                "predicted_missed": "Missed"
            }
        else:
            show_cols = ["lead_id", "channel", "message_text", "response_gap_hrs",
                         "high_intent_flag", "missed_probability", "predicted_missed"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            rename_map = {
                "lead_id": "Lead ID", "channel": "Source",
                "message_text": "Inquiry", "response_gap_hrs": "Gap (Hrs)",
                "high_intent_flag": "High Intent", "missed_probability": "Risk Score",
                "predicted_missed": "Missed"
            }

        display_df = filtered[show_cols].copy()
        if "missed_probability" in display_df.columns:
            display_df["missed_probability"] = display_df["missed_probability"].apply(lambda x: f"{x:.1%}")
        if "predicted_missed" in display_df.columns:
            display_df["predicted_missed"] = display_df["predicted_missed"].map({1: "🔴 MISSED", 0: "🟢 Safe"})
        if "high_intent_flag" in display_df.columns:
            display_df["high_intent_flag"] = display_df["high_intent_flag"].map({1: "🔥 High", 0: "Normal"})
        display_df = display_df.rename(columns=rename_map)

        st.dataframe(display_df, use_container_width=True, height=300)

        st.markdown("---")
        st.subheader("🔍 Lead Details & Smart Reply Simulator")

        selected_id = st.selectbox("Select a Lead ID:", ["-- None --"] + list(filtered["lead_id"].unique()))

        if selected_id != "-- None --":
            lead_row = filtered[filtered["lead_id"] == selected_id].iloc[0]

            d_col1, d_col2 = st.columns([1, 1])

            with d_col1:
                st.markdown("<div style='background:rgba(22,27,34,0.4); padding:1rem; border-radius:10px; border:1px solid rgba(88,166,255,0.06);'>", unsafe_allow_html=True)
                st.markdown(f"**Lead ID:** `{lead_row['lead_id']}`")

                if has_gmail_headers:
                    st.markdown(f"**Customer:** {lead_row.get('_customer_name', 'N/A')} ({lead_row.get('_customer_email', 'N/A')})")
                    st.markdown(f"**Subject:** {lead_row.get('_subject', 'N/A')}")
                    st.markdown(f"**Received:** {lead_row.get('_received_time', 'Unknown')}")
                else:
                    st.markdown(f"**Source:** {lead_row.get('channel', 'N/A')}")

                st.markdown(f"**Response Gap:** {lead_row['response_gap_hrs']:.1f} hours")
                st.markdown(f"**Risk Score:** `{lead_row['missed_probability']:.2f}`")
                st.markdown(f"**Status:** {'🔴 MISSED LEAD' if lead_row['predicted_missed'] == 1 else '🟢 Responded'}")

                st.markdown("**Original Message:**")
                st.info(lead_row["message_text"])
                st.markdown("</div>", unsafe_allow_html=True)

            with d_col2:
                st.markdown("**Smart Auto-Reply Preview:**")
                try:
                    from smart_reply_engine import generate_reply
                    reply_payload = generate_reply(
                        customer_name=lead_row.get("_customer_name") or "Valued Customer",
                        customer_email=lead_row.get("_customer_email") or "customer@example.com",
                        subject=lead_row.get("_subject") or "Enquiry",
                        message_text=lead_row["message_text"],
                        channel=lead_row.get("channel", "Email")
                    )

                    st.markdown(f"**Detected Intent:** <span class='badge badge-live' style='background:rgba(56,189,248,0.1); color:#38bdf8; border-color:rgba(56,189,248,0.2);'>{reply_payload['detected_intent'].upper()}</span>", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="email-mock">
                        <div class="email-mock-header">
                            <div>From: <span>Sales Team &lt;noreply@yourcompany.com&gt;</span></div>
                            <div>To: <span>{lead_row.get('_customer_name', 'Valued Customer')} &lt;{lead_row.get('_customer_email', 'customer@example.com')}&gt;</span></div>
                            <div>Subject: <span>{reply_payload["reply_subject"]}</span></div>
                        </div>
                        <div class="email-mock-body">{reply_payload["reply_body"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Could not load smart reply templates: {e}")
        else:
            st.info("Pick a lead ID above to drill down into details and simulate reply drafts.")

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Auto-Replies Tracker
# ══════════════════════════════════════════════════════════
elif page == "Auto-Replies Tracker":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📬 Automated Auto-Replies History")

    reply_log = load_json_log(REPLY_LOG)
    followup_status = load_json_log(FOLLOWUP_LOG)

    if not reply_log:
        st.info("No auto-replies logged yet. They are generated when the inbox monitoring detects new missed leads.")
    else:
        reply_df = pd.DataFrame(reply_log)

        st.markdown("---")
        st.subheader("Sent Reply Logs")
        show_cols = [c for c in ["lead_id", "customer_name", "customer_email", "reply_subject", "detected_intent", "replied_at"] if c in reply_df.columns]
        st.dataframe(reply_df[show_cols], use_container_width=True)

        st.markdown("---")
        st.subheader("⌛ Overdue Human Follow-Up")

        if followup_status:
            fu_items = []
            for lid, info in followup_status.items():
                if isinstance(info, dict):
                    fu_items.append({
                        "lead_id": lid,
                        "customer_name": info.get("customer_name", ""),
                        "customer_email": info.get("customer_email", ""),
                        "auto_replied_at": info.get("auto_replied_at", "—"),
                        "human_follow_up": "✅ Done" if info.get("human_followed_up") else "⌛ Pending",
                        "alert_escalated": "🔥 Overdue" if info.get("overdue_notified") else "—"
                    })

            fu_df = pd.DataFrame(fu_items)
            pending_leads = [item["lead_id"] for item in fu_items if "Pending" in item["human_follow_up"]]

            if pending_leads:
                col_sel, col_act = st.columns([2, 1])
                with col_sel:
                    action_id = st.selectbox("Mark Lead as Resolved", ["-- Select --"] + pending_leads)
                with col_act:
                    st.markdown("<div style='margin-top:1.75rem;'></div>", unsafe_allow_html=True)
                    if st.button("Complete Human Action", use_container_width=True) and action_id != "-- Select --":
                        followup_status[action_id]["human_followed_up"] = True
                        followup_status[action_id]["human_followed_up_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(FOLLOWUP_LOG, "w") as f:
                            json.dump(followup_status, f, indent=2, default=str)
                        st.success(f"Lead {action_id} marked as resolved!")
                        st.rerun()

            st.dataframe(fu_df, use_container_width=True)
        else:
            st.info("No leads awaiting manual follow-up.")

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Interactive Pipeline Graph
# ══════════════════════════════════════════════════════════
elif page == "Interactive Pipeline Graph":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🌊 Lead Flow Pipeline")

    scored = load_scored_leads()
    reply_log = load_json_log(REPLY_LOG)

    if len(scored) == 0:
        st.info("No leads available to visualize.")
    else:
        st.markdown("This interactive graph shows how leads flow through your sales pipeline.")
        total = len(scored)
        missed = int(scored["predicted_missed"].sum()) if "predicted_missed" in scored.columns else 0
        responded = total - missed

        missed_df = scored[scored["predicted_missed"] == 1] if "predicted_missed" in scored.columns else pd.DataFrame()
        high_intent_missed = int(missed_df["high_intent_flag"].sum()) if "high_intent_flag" in missed_df.columns else 0
        low_intent_missed = missed - high_intent_missed

        auto_replied = len(reply_log) if isinstance(reply_log, list) else 0
        awaiting = max(high_intent_missed - auto_replied, 0)

        if HAS_PLOTLY:
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    line=dict(color="rgba(88,166,255,0.1)", width=0.5),
                    label=["Total Leads", "Responded (Safe)", "Missed Leads",
                           "High Intent (Missed)", "Low Intent (Missed)",
                           "Auto-Replied", "Awaiting Human"],
                    color=["#38bdf8", "#34d399", "#f87171", "#fbbf24", "#484f58", "#34d399", "#f87171"]
                ),
                link=dict(
                    source=[0, 0, 2, 2, 3, 3],
                    target=[1, 2, 3, 4, 5, 6],
                    value=[max(responded, 1), max(missed, 1), max(high_intent_missed, 1),
                           max(low_intent_missed, 1), max(auto_replied, 1), max(awaiting, 1)],
                    color=["rgba(52,211,153,0.3)", "rgba(248,113,113,0.3)", "rgba(251,191,36,0.3)",
                           "rgba(72,79,88,0.3)", "rgba(52,211,153,0.3)", "rgba(248,113,113,0.3)"]
                )
            )])
            fig.update_layout(
                title_text="Customer Journey & Sales Bottlenecks",
                font_size=14, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3', height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Plotly is required to render the interactive graph.")

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Model Analytics
# ══════════════════════════════════════════════════════════
elif page == "Model Analytics":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🤖 ML Model Performance Analytics")

    # Model comparison
    if os.path.exists(MODEL_CMP):
        with open(MODEL_CMP) as f:
            cmp_data = json.load(f)

        models_dict = cmp_data.get("models", {})
        best_model = cmp_data.get("best", "N/A")

        if models_dict:
            st.markdown("#### Model AUC Comparison")
            model_names = list(models_dict.keys())
            aucs = [models_dict[m].get("auc", 0) for m in model_names]

            if HAS_PLOTLY:
                fig = px.bar(
                    x=model_names, y=aucs,
                    title="Model Test AUC Scores",
                    labels={"x": "Model", "y": "AUC Score"},
                    color=aucs,
                    color_continuous_scale=["#f87171", "#fbbf24", "#34d399", "#38bdf8", "#a78bfa"]
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#7d8590', showlegend=False,
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(88,166,255,0.06)'),
                    coloraxis_showscale=False, margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.info(f"**Best Model:** {best_model}")
    else:
        st.info("No model comparison data available. Run training first.")

    st.markdown("---")

    # Model comparison chart image
    if os.path.exists(CMP_CHART):
        st.markdown("#### Before vs After Optuna Tuning")
        st.image(CMP_CHART, use_container_width=True)

    st.markdown("---")

    # XGBoost tuning results
    if os.path.exists(XGB_TUNING):
        with open(XGB_TUNING) as f:
            xgb_data = json.load(f)
        st.markdown("#### XGBoost Tuning Results (Optuna)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best AUC", f"{xgb_data.get('best_auc', 0):.4f}")
        col2.metric("Test AUC", f"{xgb_data.get('test_auc', 0):.4f}")
        col3.metric("Trials", xgb_data.get('n_trials', 0))

        if xgb_data.get("best_params"):
            st.json(xgb_data["best_params"])

    st.markdown("---")

    # DL charts
    st.markdown("#### Deep Learning Visualizations")
    dl_cols = st.columns(3)
    for i, (path, label) in enumerate([
        (DL_HIST, "Training History"), (DL_CM, "Confusion Matrix"), (DL_ROC, "ROC Curve")
    ]):
        with dl_cols[i]:
            if os.path.exists(path) and HAS_PIL:
                st.markdown(f"**{label}**")
                st.image(path, use_container_width=True)
            else:
                st.caption(f"{label}: Not available")

    # Feature importance
    st.markdown("---")
    st.markdown("#### Feature Importance")
    if os.path.exists(FI_IMG) and HAS_PIL:
        st.image(FI_IMG, use_container_width=True)
    else:
        st.info("Feature importance chart not available.")

    # Classification report
    if os.path.exists(REPORT):
        st.markdown("---")
        st.markdown("#### Grand Ensemble Classification Report")
        with open(REPORT) as f:
            report_text = f.read()
        st.code(report_text, language="text")

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Workflow Settings
# ══════════════════════════════════════════════════════════
elif page == "Workflow Settings":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Workflow Configuration & Business Settings")
    st.markdown("Update templates, courses, placement rates, and notification boundaries. Changes save immediately.")

    save_status = st.empty()

    with st.form("settings_form"):
        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("#### Business Identity")
            company_name = st.text_input("Company Name", overrides.get("COMPANY_NAME", config.COMPANY_NAME))
            sender_name = st.text_input("Sender Name", overrides.get("SENDER_NAME", config.SENDER_NAME))
            team_phone = st.text_input("Contact Phone", overrides.get("TEAM_PHONE", config.TEAM_PHONE))
            team_email = st.text_input("Contact Email", overrides.get("TEAM_EMAIL", config.TEAM_EMAIL))
            website_url = st.text_input("Website URL", overrides.get("WEBSITE_URL", config.WEBSITE_URL))

            st.markdown("#### Escalation Rules")
            overdue_hrs = st.number_input("Overdue Alert (Hours)", value=int(overrides.get("HOURS_BEFORE_OVERDUE", config.HOURS_BEFORE_OVERDUE)), min_value=1)
            esc_hrs = st.number_input("Escalation (Hours)", value=int(overrides.get("HOURS_BEFORE_ESCALATION", config.HOURS_BEFORE_ESCALATION)), min_value=1)

        with sc2:
            st.markdown("#### Course Offerings")
            placement_rate = st.text_input("Placement Rate", overrides.get("PLACEMENT_RATE", config.PLACEMENT_RATE))
            partners = st.text_input("Hiring Partners", overrides.get("COMPANY_PARTNERS", config.COMPANY_PARTNERS))
            discount = st.text_area("Discount Text", overrides.get("DISCOUNT_INFO", config.DISCOUNT_INFO), height=80)
            scholarship = st.text_area("Scholarship Info", overrides.get("SCHOLARSHIP_INFO", config.SCHOLARSHIP_INFO), height=80)
            emi_text = st.text_area("EMI Details", overrides.get("EMI_INFO", config.EMI_INFO), height=80)

            st.markdown("#### Reply Delay (Seconds)")
            td1, td2 = st.columns(2)
            with td1:
                min_delay = st.number_input("Min Delay", value=int(overrides.get("MIN_REPLY_DELAY", config.MIN_REPLY_DELAY)), min_value=1)
            with td2:
                max_delay = st.number_input("Max Delay", value=int(overrides.get("MAX_REPLY_DELAY", config.MAX_REPLY_DELAY)), min_value=1)

        st.markdown("#### Email Signature")
        sig_val = overrides.get("EMAIL_SIGNATURE", f"Best regards,\n{sender_name}\n{company_name}\nPhone: {team_phone}\nEmail: {team_email}\nWeb: {website_url}")
        email_signature = st.text_area("Email Signature Block", sig_val, height=120)

        submitted = st.form_submit_button("Save & Update Settings", type="primary")

        if submitted:
            new_overrides = overrides.copy()
            new_overrides.update({
                "COMPANY_NAME": company_name, "SENDER_NAME": sender_name,
                "TEAM_PHONE": team_phone, "TEAM_EMAIL": team_email,
                "WEBSITE_URL": website_url, "HOURS_BEFORE_OVERDUE": overdue_hrs,
                "HOURS_BEFORE_ESCALATION": esc_hrs, "PLACEMENT_RATE": placement_rate,
                "COMPANY_PARTNERS": partners, "DISCOUNT_INFO": discount,
                "SCHOLARSHIP_INFO": scholarship, "EMI_INFO": emi_text,
                "MIN_REPLY_DELAY": min_delay, "MAX_REPLY_DELAY": max_delay,
                "EMAIL_SIGNATURE": email_signature,
            })
            save_overrides(new_overrides)
            save_status.success("Settings saved successfully!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Environment Variables Panel
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔒 Environment Variables")

    env_list = {
        "SMTP_USER": os.getenv("SMTP_USER", ""),
        "SMTP_HOST": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "IMAP_USER": os.getenv("IMAP_USER", ""),
        "NOTIFY_EMAIL": os.getenv("NOTIFY_EMAIL", ""),
        "SENDER_NAME": os.getenv("SENDER_NAME", ""),
    }

    for key, val in env_list.items():
        val_str = "❌ Not Set" if not val else f"✅ {val}"
        st.markdown(f"**{key}:** `{val_str}`")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Missed-Lead Detector &bull; AI-Powered Sales Command Center &bull; CIT Chennai Batch 2025-27
</div>
""", unsafe_allow_html=True)
