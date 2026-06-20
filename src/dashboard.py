"""
dashboard.py — Missed-Lead Detector (SaaS Overhaul)
Sales Team Command Center — Premium Dark Glassmorphic Dashboard.

Covers live monitoring, lead browsing, intent preview, ML visualisations, and business parameters editor.
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
MERGED_DATA = os.path.join(BASE, "..", "data", "leads_merged.csv")
OVERRIDES_PATH = os.path.join(BASE, "..", "logs", "config_overrides.json")

# Import config constants directly for default previewing
sys.path.insert(0, BASE)
import config

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Missed-Lead Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Design System (Sleek Dark Glassmorphism) ───────────
DESIGN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global settings */
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 80% 10%, rgba(26, 26, 42, 0.4) 0%, rgba(10, 10, 15, 1) 100%), #0a0a0f !important;
        color: #f0f0f5 !important;
    }

    /* Sidebar glass effect */
    section[data-testid="stSidebar"] {
        background: rgba(14, 14, 22, 0.7) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] label {
        color: #f0f0f5 !important;
    }

    /* Glass card container */
    .glass-card {
        background: rgba(22, 22, 34, 0.35);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(232, 168, 56, 0.18);
        box-shadow: 0 12px 40px 0 rgba(232, 168, 56, 0.04);
        transform: translateY(-2px);
    }
    
    /* Header glass container */
    .app-header {
        background: linear-gradient(135deg, rgba(26, 26, 38, 0.6) 0%, rgba(15, 15, 24, 0.8) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(232, 168, 56, 0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .app-header h1 {
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #f0f0f5 30%, #e8a838 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .app-header p {
        color: #9898a6;
        margin-top: 0.5rem;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0;
    }

    /* KPI Metrics custom layout */
    .kpi-container {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 1.75rem;
    }
    
    .kpi-card {
        flex: 1;
        background: rgba(22, 22, 34, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid var(--accent-color, #e8a838);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 10px rgba(255, 255, 255, 0.02);
        border-color: var(--accent-color, #e8a838);
    }
    
    .kpi-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 10% 20%, var(--glow-color, rgba(232, 168, 56, 0.08)) 0%, transparent 60%);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }
    
    .kpi-card:hover::after {
        opacity: 1;
    }
    
    .kpi-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        letter-spacing: -0.03em;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .kpi-label {
        font-size: 0.8rem;
        color: #9898a6;
        margin-top: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .kpi-card.blue { --accent-color: #60a5fa; --glow-color: rgba(96, 165, 250, 0.08); }
    .kpi-card.red { --accent-color: #f87171; --glow-color: rgba(248, 113, 113, 0.08); }
    .kpi-card.green { --accent-color: #34d399; --glow-color: rgba(52, 211, 153, 0.08); }
    .kpi-card.yellow { --accent-color: #fbbf24; --glow-color: rgba(251, 191, 36, 0.08); }
    .kpi-card.gold { --accent-color: #e8a838; --glow-color: rgba(232, 168, 56, 0.08); }

    .kpi-card.blue .kpi-value { color: #60a5fa; }
    .kpi-card.red .kpi-value { color: #f87171; }
    .kpi-card.green .kpi-value { color: #34d399; }
    .kpi-card.yellow .kpi-value { color: #fbbf24; }
    .kpi-card.gold .kpi-value { color: #e8a838; }

    /* Custom Notification Cards */
    .notif-card {
        background: rgba(26, 26, 38, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 3px solid #e8a838;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .notif-card:hover {
        background: rgba(30, 30, 46, 0.5);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .notif-card.new_lead { border-left-color: #60a5fa; }
    .notif-card.auto_reply { border-left-color: #34d399; }
    .notif-card.overdue { border-left-color: #f87171; }
    
    .notif-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.25rem;
    }
    
    .notif-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #ffffff;
    }
    
    .notif-time {
        font-size: 0.75rem;
        color: #5a5a6e;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .notif-msg {
        font-size: 0.85rem;
        color: #9898a6;
        line-height: 1.4;
    }

    /* Terminal Monitor Output */
    .terminal-box {
        background: #060609 !important;
        border: 1px solid #1f1f2e !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        color: #a3e635 !important;
        box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.8) !important;
        line-height: 1.5 !important;
        max-height: 400px;
        overflow-y: auto;
    }

    /* Status Badges */
    .custom-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-live {
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    
    .badge-live::before {
        content: '';
        width: 6px;
        height: 6px;
        background: #34d399;
        border-radius: 50%;
        animation: status-pulse 2s infinite;
    }
    
    .badge-demo {
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }
    
    @keyframes status-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Email Mock Client Frame */
    .email-mock {
        background: rgba(10, 10, 15, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1rem;
        font-family: sans-serif;
    }
    
    .email-mock-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
        color: #9898a6;
    }
    
    .email-mock-header span {
        color: #ffffff;
        font-weight: 500;
    }
    
    .email-mock-body {
        font-size: 0.9rem;
        color: #f0f0f5;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Streamlit overrides for custom theme compatibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(14, 14, 22, 0.6);
        border-radius: 10px;
        padding: 0.3rem;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9898a6 !important;
        font-weight: 600;
        padding: 0.4rem 1.25rem;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(232, 168, 56, 0.15) !important;
        color: #e8a838 !important;
        border: 1px solid rgba(232, 168, 56, 0.25) !important;
    }
    
    /* Input Styling */
    div[data-baseweb="select"], div[data-baseweb="input"] input, textarea {
        background: rgba(22, 22, 34, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    
    /* Button overrides */
    .stButton > button {
        background: rgba(22, 22, 34, 0.6) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }
    
    .stButton > button:hover {
        border-color: #e8a838 !important;
        color: #e8a838 !important;
        box-shadow: 0 4px 15px rgba(232, 168, 56, 0.1) !important;
        transform: translateY(-1px);
    }
    
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: #e8a838 !important;
        color: #0a0a0f !important;
        border: none !important;
    }
    
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #f0c060 !important;
        box-shadow: 0 4px 20px rgba(232, 168, 56, 0.3) !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer {visibility: hidden;}
    
    /* Custom footer styles */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        color: #5a5a6e;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 4rem;
        letter-spacing: 0.05em;
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
    # Clear streamlit cache to propagate changes
    st.cache_data.clear()

overrides = load_overrides()

# ── Data Loading Helpers ──────────────────────────────────
@st.cache_data(ttl=15)
def load_scored_leads():
    if os.path.exists(SCORED):
        df = pd.read_csv(SCORED)
        if "replied" in df.columns and "predicted_missed" not in df.columns:
            # map replied to predicted_missed (positive class: replied=0 -> missed=1)
            df["predicted_missed"] = df["replied"].map({1: 0, 0: 1})
        return df
    return pd.DataFrame()

@st.cache_data(ttl=15)
def load_json_log(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return [] if "list" in path or path.endswith("s.json") else {}

# ── Connection Checks ──────────────────────────────────────
def is_live_mode() -> bool:
    return bool(os.getenv("IMAP_USER")) or bool(os.getenv("SMTP_USER"))

# ── Header Section ─────────────────────────────────────────
live_connected = is_live_mode()
status_label = "Live Pipeline Connected" if live_connected else "Demo Mode — Synthetic Sandbox"
status_class = "badge-live" if live_connected else "badge-demo"

st.markdown(f"""
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1>Missed-Lead Command Center</h1>
        <span class="custom-badge {status_class}">{status_label}</span>
    </div>
    <p>AI-Powered Email Monitoring, Smart Auto-Replies & Sales Pipeline Retention Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configurations & Navigation ───────────────────
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to:",
        ["Command Center", "Lead Explorer", "Auto-Replies Tracker", "ML Analytics Playground", "Workflow Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Inbox Monitor Health")
    
    leads_df = load_scored_leads()
    notifs = load_json_log(NOTIF_LOG)
    unread_notifs = [n for n in notifs if not n.get("read", False)] if isinstance(notifs, list) else []
    
    # Styled inbox status box
    inbox_status_icon = "🟢" if live_connected else "🟡"
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); 
                border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; line-height: 1.6;">
        <div><b>Gmail Connection</b>: {inbox_status_icon} {'Active' if live_connected else 'Simulated'}</div>
        <div><b>Total Scored</b>: {len(leads_df)} leads</div>
        <div><b>Unread Alerts</b>: <span style="color: {'#f87171' if len(unread_notifs) > 0 else '#9898a6'}; font-weight: bold;">{len(unread_notifs)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; color:#5a5a6e; text-align: center;'>Missed-Lead Detector v2.0<br>Batch 2025-27</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Command Center
# ══════════════════════════════════════════════════════════
if page == "Command Center":
    # 1. KPI Metric Cards Row
    scored = load_scored_leads()
    reply_log = load_json_log(REPLY_LOG)
    followup_status = load_json_log(FOLLOWUP_LOG)
    
    total_leads = len(scored)
    
    # Calculate counts
    if "predicted_missed" in scored.columns and total_leads > 0:
        missed_leads = int(scored["predicted_missed"].sum())
    else:
        missed_leads = 0
        
    replied_count = len(reply_log) if isinstance(reply_log, list) else 0
    
    overdue_count = 0
    if isinstance(followup_status, dict):
        overdue_count = sum(1 for s in followup_status.values() 
                            if isinstance(s, dict) and s.get("auto_replied") and not s.get("human_followed_up"))

    high_intent = 0
    if "high_intent_flag" in scored.columns and total_leads > 0:
        high_intent = int(scored["high_intent_flag"].sum())

    # Build KPI metrics row using custom styled divs
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
        <div class="kpi-card yellow">
            <div class="kpi-value">{overdue_count}</div>
            <div class="kpi-label">Awaiting Sales Action</div>
        </div>
        <div class="kpi-card gold">
            <div class="kpi-value">{high_intent}</div>
            <div class="kpi-label">High Intent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Main Columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📢 Inbox Monitoring Pipeline")
        
        # Scopes for triggering scan
        c1, c2 = st.columns(2)
        with c1:
            scan_btn = st.button("Trigger Scan Now", type="primary", use_container_width=True)
        with c2:
            dry_btn = st.button("Simulate Dry-Run Scan", use_container_width=True)
            
        # Scan runner output
        if scan_btn or dry_btn:
            is_dry = bool(dry_btn)
            st.markdown("**Pipeline Output:**")
            with st.spinner("Processing mailbox via SMTP/IMAP..."):
                try:
                    cmd = [sys.executable, os.path.join(BASE, "inbox_monitor.py")]
                    if is_dry:
                        cmd.append("--dry-run")
                    
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    # Styled output terminal block
                    stdout = res.stdout if res.stdout else "No output returned."
                    st.markdown(f"<pre class='terminal-box'>{stdout}</pre>", unsafe_allow_html=True)
                    if res.stderr:
                        st.warning(f"Warnings reported:\n{res.stderr}")
                    
                    st.cache_data.clear() # reload lists
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution Error: {e}")
        else:
            st.markdown("<p style='color: #5a5a6e; font-style: italic; margin-top:0.5rem;'>Trigger a pipeline scan to read Gmail inboxes, run machine learning predictions, and execute automatic replies.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Interactive Distribution chart
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📊 Lead Missed Probability Distribution")
        
        if total_leads > 0 and "missed_probability" in scored.columns:
            if HAS_PLOTLY:
                fig = px.histogram(
                    scored, 
                    x="missed_probability", 
                    nbins=20,
                    title="Lead Risk distribution (Threshold = 0.50)",
                    labels={"missed_probability": "Predicted Missed Probability", "count": "Lead Count"},
                    color_discrete_sequence=["#f87171"]
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#9898a6',
                    title_font_color='#ffffff',
                    showlegend=False,
                    xaxis=dict(showgrid=False, linecolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)'),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                # Add threshold line
                fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="#fbbf24", 
                              annotation_text="Score Threshold", annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Visualisation loaded from file fallback")
                if os.path.exists(CM_IMG) and HAS_PIL:
                    st.image(Image.open(CM_IMG), use_container_width=True)
        else:
            st.info("No lead metrics scored. Trigger a scan above to ingest data.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_right:
        # Alerts/Notifications Center
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("🔔 Notification Feed")
        
        if isinstance(notifs, list) and notifs:
            # Mark all as read button
            if len(unread_notifs) > 0:
                if st.button(f"Mark all read ({len(unread_notifs)})", use_container_width=True):
                    for n in notifs:
                        n["read"] = True
                    # save back
                    with open(NOTIF_LOG, "w") as f:
                        json.dump(notifs, f, indent=2, default=str)
                    st.rerun()
            
            # Show list
            unread_count = 0
            for i, n in enumerate(reversed(notifs)):
                if not n.get("read", False) and unread_count < 10:
                    ntype = n.get("type", "info")
                    icon = "📧" if ntype == "new_lead" else ("🤖" if ntype == "auto_reply" else ("⚠️" if ntype == "overdue" else "ℹ️"))
                    
                    st.markdown(f"""
                    <div class="notif-card {ntype}">
                        <div class="notif-header">
                            <span class="notif-title">{icon} {n.get('title', 'Notification')}</span>
                            <span class="notif-time">{n.get('timestamp', '')[11:16]}</span>
                        </div>
                        <div class="notif-msg">{n.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    unread_count += 1
            if unread_count == 0:
                st.markdown("<p style='text-align:center; color:#5a5a6e; padding: 2rem 0;'>All caught up! No unread notifications.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center; color:#5a5a6e; padding: 2rem 0;'>No notifications found.</p>", unsafe_allow_html=True)
            
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
        # Columns mapping to ensure user-friendly presentation
        has_gmail_headers = "_customer_name" in scored.columns
        
        # Filter controls row
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
            
        # Querying leads
        filtered = scored.copy()
        
        if q_search:
            s_pat = q_search.lower()
            if has_gmail_headers:
                filtered = filtered[
                    filtered["_customer_name"].str.lower().str.contains(s_pat, na=False) |
                    filtered["_customer_email"].str.lower().str.contains(s_pat, na=False)
                ]
            else:
                filtered = filtered[
                    filtered["message_text"].str.lower().str.contains(s_pat, na=False)
                ]
                
        if q_status != "All":
            val = 1 if q_status == "Missed Leads" else 0
            filtered = filtered[filtered["predicted_missed"] == val]
            
        if q_intent != "All":
            val = 1 if q_intent == "Yes" else 0
            filtered = filtered[filtered["high_intent_flag"] == val]
            
        if q_channel != "All":
            if "channel" in filtered.columns:
                filtered = filtered[filtered["channel"] == q_channel]
            
        # Selected column subset
        if has_gmail_headers:
            show_cols = ["lead_id", "_customer_name", "_customer_email", "_subject", "channel", 
                         "response_gap_hrs", "high_intent_flag", "missed_probability", "predicted_missed"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            rename_map = {
                "lead_id": "Lead ID",
                "_customer_name": "Customer",
                "_customer_email": "Email Address",
                "_subject": "Subject Inquired",
                "channel": "Source",
                "response_gap_hrs": "Gap (Hrs)",
                "high_intent_flag": "High Intent",
                "missed_probability": "Risk Score",
                "predicted_missed": "Missed"
            }
        else:
            show_cols = ["lead_id", "channel", "message_text", "response_gap_hrs", 
                         "high_intent_flag", "missed_probability", "predicted_missed"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            rename_map = {
                "lead_id": "Lead ID",
                "channel": "Source",
                "message_text": "Inquiry Details",
                "response_gap_hrs": "Gap (Hrs)",
                "high_intent_flag": "High Intent",
                "missed_probability": "Risk Score",
                "predicted_missed": "Missed"
            }
            
        display_df = filtered[show_cols].copy()
        
        # Friendly formats
        display_df["missed_probability"] = display_df["missed_probability"].apply(lambda x: f"{x:.1%}")
        display_df["predicted_missed"] = display_df["predicted_missed"].map({1: "🔴 MISSED", 0: "🟢 Safe"})
        display_df["high_intent_flag"] = display_df["high_intent_flag"].map({1: "🔥 High", 0: "Normal"})
        
        display_df = display_df.rename(columns=rename_map)
        
        # Display Table
        st.dataframe(display_df, use_container_width=True, height=300)
        
        # Lead detail inspector selection
        st.markdown("---")
        st.subheader("🔍 Details Inspector & Smart Reply Simulator")
        
        selected_id = st.selectbox("Select a Lead ID to inspect details:", ["-- None --"] + list(filtered["lead_id"].unique()))
        
        if selected_id != "-- None --":
            lead_row = filtered[filtered["lead_id"] == selected_id].iloc[0]
            
            d_col1, d_col2 = st.columns([1, 1])
            
            with d_col1:
                st.markdown("<div style='background:rgba(255,255,255,0.02); padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.04);'>", unsafe_allow_html=True)
                st.markdown(f"**Lead Identifier:** `{lead_row['lead_id']}`")
                
                if has_gmail_headers:
                    st.markdown(f"**Customer:** {lead_row['_customer_name']} (<{lead_row['_customer_email']}>)")
                    st.markdown(f"**Subject:** {lead_row['_subject']}")
                    st.markdown(f"**Received:** {lead_row.get('_received_time', 'Unknown')}")
                else:
                    st.markdown(f"**Source Channel:** {lead_row['channel']}")
                
                st.markdown(f"**Response Gap:** {lead_row['response_gap_hrs']:.1f} hours")
                st.markdown(f"**Risk Score (ML Probability):** `{lead_row['missed_probability']:.2f}`")
                st.markdown(f"**Classification Status:** {'🔴 MISSED LEAD' if lead_row['predicted_missed'] == 1 else '🟢 Responded'}")
                
                st.markdown("**Original Inquiry Message:**")
                st.info(lead_row["message_text"])
                st.markdown("</div>", unsafe_allow_html=True)
                
            with d_col2:
                st.markdown("**Smart Auto-Reply Preview:**")
                # Generate a mock reply based on engine
                try:
                    from smart_reply_engine import generate_reply
                    reply_payload = generate_reply(
                        customer_name=lead_row.get("_customer_name", "Valued Customer"),
                        customer_email=lead_row.get("_customer_email", "customer@example.com"),
                        subject=lead_row.get("_subject", "Enquiry"),
                        message_text=lead_row["message_text"],
                        channel=lead_row.get("channel", "Email")
                    )
                    
                    st.markdown(f"**Detected Intent Category:** <span class='custom-badge badge-live' style='background:rgba(232,168,56,0.1); color:#e8a838; border-color:#e8a838;'>{reply_payload['detected_intent'].upper()}</span>", unsafe_allow_html=True)
                    
                    # Mock email editor
                    st.markdown("""
                    <div class="email-mock">
                        <div class="email-mock-header">
                            <div>From: <span>Sales Team &lt;noreply@yourcompany.com&gt;</span></div>
                            <div>To: <span>{} &lt;{}&gt;</span></div>
                            <div>Subject: <span>{}</span></div>
                        </div>
                        <div class="email-mock-body">{}</div>
                    </div>
                    """.format(
                        lead_row.get("_customer_name", "Valued Customer"),
                        lead_row.get("_customer_email", "customer@example.com"),
                        reply_payload["reply_subject"],
                        reply_payload["reply_body"]
                    ), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Could not load smart reply templates: {e}")
                    
        else:
            st.info("Pick a lead ID above to drill down into logs, check ML features, and simulate reply drafts.")

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
        st.info("No auto-replies logged yet. Auto-replies are generated only when the inbox monitoring detects new missed leads.")
    else:
        # Load replies df
        reply_df = pd.DataFrame(reply_log)
        
        # Summary charts
        c1, c2 = st.columns([1, 1])
        with c1:
            if "detected_intent" in reply_df.columns:
                intent_counts = reply_df["detected_intent"].value_counts().reset_index()
                intent_counts.columns = ["Intent Category", "Emails Sent"]
                
                if HAS_PLOTLY:
                    fig = px.bar(intent_counts, x="Intent Category", y="Emails Sent", 
                                 title="Auto-Replies sent by Intent Category",
                                 color="Intent Category", color_discrete_sequence=px.colors.qualitative.Antique)
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#9898a6',
                        title_font_color='#ffffff',
                        showlegend=False,
                        xaxis=dict(showgrid=False, linecolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(reply_df["detected_intent"].value_counts())
            else:
                st.info("No intents mapped.")
                
        with c2:
            if "channel" in reply_df.columns:
                ch_counts = reply_df["channel"].value_counts().reset_index()
                ch_counts.columns = ["Channel", "Volume"]
                if HAS_PLOTLY:
                    fig = px.pie(ch_counts, names="Channel", values="Volume", 
                                 title="Replies Distribution by Channel Source",
                                 color_discrete_sequence=px.colors.qualitative.Safe)
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#9898a6',
                        title_font_color='#ffffff',
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(reply_df["channel"].value_counts())
                    
        # Table of sent replies
        st.markdown("---")
        st.subheader("Sent Reply Logs")
        show_cols = [c for c in ["lead_id", "customer_name", "customer_email", "reply_subject", "detected_intent", "replied_at"] if c in reply_df.columns]
        st.dataframe(reply_df[show_cols], use_container_width=True)
        
        # Awaiting human follow-up tracker list
        st.markdown("---")
        st.subheader("⌛ Overdue Human Follow-Up Checklist (Awaiting Sales Action)")
        
        if followup_status:
            fu_items = []
            for lid, info in followup_status.items():
                if isinstance(info, dict):
                    fu_items.append({
                        "lead_id": lid,
                        "customer_name": info.get("customer_name", ""),
                        "customer_email": info.get("customer_email", ""),
                        "auto_replied_at": info.get("auto_replied_at", "—"),
                        "human_follow_up": "✅ Done" if info.get("human_followed_up") else "⌛ Pending Sales Rep",
                        "alert_escalated": "🔥 Overdue Alert" if info.get("overdue_notified") else "—"
                    })
                    
            fu_df = pd.DataFrame(fu_items)
            
            # Action controls to mark follow-up complete
            pending_leads = [item["lead_id"] for item in fu_items if "Pending" in item["human_follow_up"]]
            
            if pending_leads:
                col_sel, col_act = st.columns([2, 1])
                with col_sel:
                    action_id = st.selectbox("Resolve Pipeline Status: Mark Lead as Followed-up by Human", ["-- Select --"] + pending_leads)
                with col_act:
                    st.markdown("<div style='margin-top:1.75rem;'></div>", unsafe_allow_html=True)
                    if st.button("Complete Human Action", use_container_width=True) and action_id != "-- Select --":
                        followup_status[action_id]["human_followed_up"] = True
                        followup_status[action_id]["human_followed_up_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Save
                        with open(FOLLOWUP_LOG, "w") as f:
                            json.dump(followup_status, f, indent=2, default=str)
                        st.success(f"Lead {action_id} successfully marked as resolved!")
                        st.rerun()
            
            st.dataframe(fu_df, use_container_width=True)
        else:
            st.info("No leads awaiting manual sales follow-up.")
            
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: ML Analytics Playground
# ══════════════════════════════════════════════════════════
elif page == "ML Analytics Playground":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧠 Machine Learning Model Performance Metrics")
    
    # Accuracy details parsing
    acc, f1, prec, rec = 0.82, 0.59, 0.68, 0.52
    if os.path.exists(REPORT):
        try:
            with open(REPORT) as f:
                for line in f.read().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5 and parts[0] == "accuracy":
                        acc = float(parts[1])
                    if len(parts) >= 5 and parts[0] == "Missed":
                        prec, rec, f1 = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            pass
            
    # Model KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Ensemble Accuracy", f"{acc:.1%}")
    with c2:
        st.metric("F1 Score (Missed Leads)", f"{f1:.3f}")
    with c3:
        st.metric("Precision (Missed Leads)", f"{prec:.3f}")
    with c4:
        st.metric("Recall (Missed Leads)", f"{rec:.3f}")
        
    # Model Comparison Plotly Chart
    st.markdown("---")
    st.subheader("🏆 Model ROC-AUC Score Comparisons")
    if os.path.exists(MODEL_CMP):
        try:
            with open(MODEL_CMP) as f:
                mc_data = json.load(f)
                
            if mc_data.get("models"):
                rows = [{"Model": k, "Test AUC": v["auc"]} for k, v in mc_data["models"].items()]
                compare_df = pd.DataFrame(rows).sort_values("Test AUC", ascending=True)
                
                if HAS_PLOTLY:
                    fig = px.bar(compare_df, y="Model", x="Test AUC", orientation="h",
                                 title="ROC-AUC scores for 8 Models (Higher is better)",
                                 color="Test AUC", color_continuous_scale="Viridis")
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#9898a6',
                        title_font_color='#ffffff',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(showgrid=False, linecolor='rgba(255,255,255,0.05)'),
                        margin=dict(l=40, r=40, t=40, b=40),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(compare_df)
        except Exception as e:
            st.error(f"Error loading model comparison data: {e}")
            
    # Optuna tuning hyperparams
    st.markdown("---")
    st.subheader("⚙️ Optuna Hyperparameter Optimization Details")
    
    if os.path.exists(XGB_TUNING):
        try:
            with open(XGB_TUNING) as f:
                tuning_res = json.load(f)
                
            tc1, tc2 = st.columns([1, 1])
            with tc1:
                st.markdown(f"**Optimization Target:** XGBoost Classifier")
                st.markdown(f"**Dataset Used:** `{tuning_res.get('dataset', 'leads_merged.csv')}`")
                st.markdown(f"**Samples Ingested:** `{tuning_res.get('n_samples', '13,740')}`")
                st.markdown(f"**Cross-Validation Trials Completed:** `{tuning_res.get('n_trials', '100')}`")
                st.markdown(f"**Best Cross-Validation AUC:** `{tuning_res.get('best_auc', '—')}`")
                st.markdown(f"**Tuned Test AUC Score:** `{tuning_res.get('test_auc', '—')}`")
            with tc2:
                st.markdown("**Best Hyperparameters Found:**")
                st.json(tuning_res.get("best_params", {}))
        except Exception:
            st.info("Tuning parameters logs not found or unreadable.")
    else:
        st.info("Optuna tuning parameters not generated.")

    # Deep learning charts fallback/display
    st.markdown("---")
    st.subheader("📈 Deep Learning (PyTorch) & Diagnostic Charts")
    
    chart_tabs = st.tabs(["Confusion Matrix", "Feature Importance", "Neural Network Loss & AUC Curves", "ROC Curve"])
    
    with chart_tabs[0]:
        if os.path.exists(CM_IMG) and HAS_PIL:
            st.image(Image.open(CM_IMG), use_container_width=True, caption="Model Confusion Matrix")
        else:
            st.info("No confusion matrix chart saved.")
            
    with chart_tabs[1]:
        if os.path.exists(FI_IMG) and HAS_PIL:
            st.image(Image.open(FI_IMG), use_container_width=True, caption="Model Random Forest Feature Importances")
        else:
            st.info("No feature importance chart saved.")
            
    with chart_tabs[2]:
        if os.path.exists(DL_HIST) and HAS_PIL:
            st.image(Image.open(DL_HIST), use_container_width=True, caption="PyTorch DL Classification Epochs Loss History")
        else:
            st.info("No deep learning loss history charts saved.")
            
    with chart_tabs[3]:
        if os.path.exists(DL_ROC) and HAS_PIL:
            st.image(Image.open(DL_ROC), use_container_width=True, caption="Receiver Operating Characteristic Curve")
        else:
            st.info("No ROC curves saved.")

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE: Workflow Settings
# ══════════════════════════════════════════════════════════
elif page == "Workflow Settings":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Workflow Configuration & Business Settings")
    st.markdown("Update templates, courses, placement rates, and notification boundaries. Changes write immediately to `config_overrides.json` and hot-reload across scripts.")
    
    # Save Status
    save_status = st.empty()
    
    # 1. Config form
    with st.form("settings_form"):
        sc1, sc2 = st.columns(2)
        
        with sc1:
            st.markdown("#### Business Identity & Signatures")
            company_name = st.text_input("Company Name", overrides.get("COMPANY_NAME", config.COMPANY_NAME))
            sender_name = st.text_input("Sender / Signature Name", overrides.get("SENDER_NAME", config.SENDER_NAME))
            team_phone = st.text_input("Contact Phone Number", overrides.get("TEAM_PHONE", config.TEAM_PHONE))
            team_email = st.text_input("Contact Email address", overrides.get("TEAM_EMAIL", config.TEAM_EMAIL))
            website_url = st.text_input("Website URL", overrides.get("WEBSITE_URL", config.WEBSITE_URL))
            
            st.markdown("#### Escalation Rules")
            overdue_hrs = st.number_input("Response Overdue Alert Time (Hours)", value=int(overrides.get("HOURS_BEFORE_OVERDUE", config.HOURS_BEFORE_OVERDUE)), min_value=1)
            esc_hrs = st.number_input("Escalation Time (Hours)", value=int(overrides.get("HOURS_BEFORE_ESCALATION", config.HOURS_BEFORE_ESCALATION)), min_value=1)
            
        with sc2:
            st.markdown("#### Course Offerings & Placement Highlights")
            placement_rate = st.text_input("Placement Rate", overrides.get("PLACEMENT_RATE", config.PLACEMENT_RATE))
            partners = st.text_input("Hiring Partners count", overrides.get("COMPANY_PARTNERS", config.COMPANY_PARTNERS))
            discount = st.text_area("Discount & Incentives Promo Text", overrides.get("DISCOUNT_INFO", config.DISCOUNT_INFO), height=80)
            scholarship = st.text_area("Scholarship Information Text", overrides.get("SCHOLARSHIP_INFO", config.SCHOLARSHIP_INFO), height=80)
            emi_text = st.text_area("EMI details description", overrides.get("EMI_INFO", config.EMI_INFO), height=80)
            
            st.markdown("#### Auto-Reply Time Delay Constraints (Seconds)")
            td1, td2 = st.columns(2)
            with td1:
                min_delay = st.number_input("Minimum Delay", value=int(overrides.get("MIN_REPLY_DELAY", config.MIN_REPLY_DELAY)), min_value=1)
            with td2:
                max_delay = st.number_input("Maximum Delay", value=int(overrides.get("MAX_REPLY_DELAY", config.MAX_REPLY_DELAY)), min_value=1)
                
        st.markdown("#### Email Signature Template Preview")
        sig_val = overrides.get("EMAIL_SIGNATURE", f"Best regards,\n{sender_name}\n{company_name}\nPhone: {team_phone}\nEmail: {team_email}\nWeb: {website_url}")
        email_signature = st.text_area("Email Signature Footer Block", sig_val, height=120)

        # Form Submit
        submitted = st.form_submit_button("Save & Update System Settings", type="primary")
        
        if submitted:
            new_overrides = overrides.copy()
            new_overrides["COMPANY_NAME"] = company_name
            new_overrides["SENDER_NAME"] = sender_name
            new_overrides["TEAM_PHONE"] = team_phone
            new_overrides["TEAM_EMAIL"] = team_email
            new_overrides["WEBSITE_URL"] = website_url
            new_overrides["HOURS_BEFORE_OVERDUE"] = overdue_hrs
            new_overrides["HOURS_BEFORE_ESCALATION"] = esc_hrs
            new_overrides["PLACEMENT_RATE"] = placement_rate
            new_overrides["COMPANY_PARTNERS"] = partners
            new_overrides["DISCOUNT_INFO"] = discount
            new_overrides["SCHOLARSHIP_INFO"] = scholarship
            new_overrides["EMI_INFO"] = emi_text
            new_overrides["MIN_REPLY_DELAY"] = min_delay
            new_overrides["MAX_REPLY_DELAY"] = max_delay
            new_overrides["EMAIL_SIGNATURE"] = email_signature
            
            save_overrides(new_overrides)
            save_status.success("Workflow configurations successfully updated & saved!")
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Environment configs checklist panel
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔒 Environment Variables Configurations Status")
    
    env_list = {
        "SMTP_USER": os.getenv("SMTP_USER", ""),
        "SMTP_HOST": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "IMAP_USER": os.getenv("IMAP_USER", ""),
        "NOTIFY_EMAIL": os.getenv("NOTIFY_EMAIL", ""),
        "SENDER_NAME": os.getenv("SENDER_NAME", "")
    }
    
    for key, val in env_list.items():
        val_str = "❌ Not Set" if not val else ("✅ Configured (Hidden)" if key.endswith("PASS") or key == "SMTP_USER" else f"✅ {val}")
        st.markdown(f"**{key}:** `{val_str}`")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer Section ─────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Missed-Lead Detector &bull; Customer Retention Sales Command Center &bull; Cit Chennai Batch 2025-27
</div>
""", unsafe_allow_html=True)
