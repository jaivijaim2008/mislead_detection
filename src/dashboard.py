"""
dashboard.py — Missed-Lead Detector
Sales Team Command Center — Streamlit dashboard.

Shows: inbox monitoring, auto-reply log, follow-up status, notifications, model performance.

Run: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, json, subprocess, sys
from datetime import datetime, timezone

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

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Missed-Lead Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System ──────────────────────────────────────────
# Color palette - Sophisticated dark theme with warm accent
COLORS = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_card": "#1a1a24",
    "bg_card_hover": "#22222e",
    "bg_elevated": "#252532",
    "accent": "#e8a838",       # Warm amber/gold
    "accent_soft": "#f0c060",
    "accent_muted": "rgba(232, 168, 56, 0.15)",
    "success": "#34d399",
    "success_muted": "rgba(52, 211, 153, 0.15)",
    "danger": "#f87171",
    "danger_muted": "rgba(248, 113, 113, 0.15)",
    "warning": "#fbbf24",
    "warning_muted": "rgba(251, 191, 36, 0.15)",
    "info": "#60a5fa",
    "info_muted": "rgba(96, 165, 250, 0.15)",
    "text_primary": "#f0f0f5",
    "text_secondary": "#9898a6",
    "text_muted": "#5a5a6e",
    "border": "#2a2a38",
    "border_light": "#35354a",
}

# ── Custom CSS ─────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {{
        --bg-primary: {COLORS["bg_primary"]};
        --bg-secondary: {COLORS["bg_secondary"]};
        --bg-card: {COLORS["bg_card"]};
        --bg-card-hover: {COLORS["bg_card_hover"]};
        --accent: {COLORS["accent"]};
        --accent-soft: {COLORS["accent_soft"]};
        --text-primary: {COLORS["text_primary"]};
        --text-secondary: {COLORS["text_secondary"]};
        --text-muted: {COLORS["text_muted"]};
        --border: {COLORS["border"]};
    }}

    /* Global typography */
    html, body, [class*="st-"] {{
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* Main background */
    .stApp {{
        background: var(--bg-primary);
    }}

    /* Main content text visibility */
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {{
        color: var(--text-primary) !important;
    }}

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {{
        color: var(--text-primary) !important;
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }}

    /* Sidebar text visibility */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio span,
    section[data-testid="stSidebar"] [data-baseweb="radio"] span,
    section[data-testid="stSidebar"] [data-baseweb="radio"] label,
    section[data-testid="stSidebar"] span {{
        color: var(--text-primary) !important;
    }}

    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{
        color: var(--text-primary) !important;
    }}

    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div {{
        color: var(--text-primary) !important;
    }}

    /* Header */
    .main-header {{
        background: linear-gradient(135deg, {COLORS["bg_card"]} 0%, {COLORS["bg_elevated"]} 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }}

    .main-header::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, {COLORS["accent_muted"]} 0%, transparent 70%);
        pointer-events: none;
    }}

    .main-header h1 {{
        color: var(--text-primary);
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        position: relative;
    }}

    .main-header p {{
        color: var(--text-secondary);
        margin: 0.75rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        position: relative;
    }}

    /* Metric cards - Premium glass effect */
    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }}

    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--card-accent, var(--accent)), transparent);
        opacity: 0;
        transition: opacity 0.2s ease;
    }}

    .metric-card:hover {{
        border-color: var(--border-light);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }}

    .metric-card:hover::before {{
        opacity: 1;
    }}

    .metric-value {{
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.04em;
    }}

    .metric-label {{
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Card accent variants */
    .metric-card.accent-danger {{ --card-accent: var(--danger); }}
    .metric-card.accent-success {{ --card-accent: var(--success); }}
    .metric-card.accent-warning {{ --card-accent: var(--warning); }}
    .metric-card.accent-info {{ --card-accent: var(--info); }}
    .metric-card.accent-gold {{ --card-accent: var(--accent); }}

    .metric-card.accent-danger .metric-value {{ color: var(--danger); }}
    .metric-card.accent-success .metric-value {{ color: var(--success); }}
    .metric-card.accent-warning .metric-value {{ color: var(--warning); }}
    .metric-card.accent-info .metric-value {{ color: var(--info); }}
    .metric-card.accent-gold .metric-value {{ color: var(--accent); }}

    /* Section headers */
    .section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1.25rem 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* Streamlit containers */
    .stAlert p,
    .stAlert span,
    .stAlert div {{
        color: var(--text-primary) !important;
    }}

    /* Dataframe cells */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] span {{
        color: var(--text-primary) !important;
    }}

    /* Code blocks */
    .stCode code,
    .stCode pre,
    code, pre {{
        color: var(--text-primary) !important;
    }}

    .section-header::before {{
        content: '';
        width: 3px;
        height: 1.1rem;
        background: var(--accent);
        border-radius: 2px;
    }}

    /* Status badges */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }}

    .status-live {{
        background: var(--success_muted);
        color: var(--success);
        border: 1px solid rgba(52, 211, 153, 0.2);
    }}

    .status-live::before {{
        content: '';
        width: 6px;
        height: 6px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }}

    .status-demo {{
        background: var(--warning_muted);
        color: var(--warning);
        border: 1px solid rgba(251, 191, 36, 0.2);
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    /* Notification cards */
    .notif-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--danger);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }}

    .notif-card:hover {{
        border-color: var(--border-light);
        background: var(--bg-card-hover);
    }}

    .notif-card.auto-reply {{
        border-left-color: var(--success);
    }}

    .notif-card.overdue {{
        border-left-color: var(--warning);
    }}

    .notif-title {{
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }}

    .notif-time {{
        color: var(--text-muted);
        font-size: 0.75rem;
    }}

    .notif-message {{
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }}

    /* Empty state */
    .empty-state {{
        text-align: center;
        padding: 4rem 2rem;
        color: var(--text-secondary);
    }}

    .empty-state-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.7;
    }}

    .empty-state-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }}

    .empty-state-text {{
        font-size: 0.9rem;
        max-width: 400px;
        margin: 0 auto;
        color: var(--text-primary);
    }}

    /* Button styling */
    .stButton > button {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        border-color: var(--accent) !important;
        background: var(--bg-card-hover) !important;
    }}

    .stButton > button:focus {{
        box-shadow: 0 0 0 2px var(--accent_muted) !important;
        border-color: var(--accent) !important;
    }}

    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: var(--accent) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        font-weight: 600 !important;
    }}

    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: var(--accent-soft) !important;
    }}

    /* Dataframe styling */
    .stDataFrame {{
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Metric component */
    [data-testid="stMetric"] {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-variant-numeric: tabular-nums;
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--text-secondary) !important;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: var(--bg-secondary);
        border-radius: 8px;
        padding: 0.25rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.5rem 1rem;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }}

    /* Focus states for accessibility */
    *:focus-visible {{
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--border-light);
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem 0;
        color: var(--text-secondary);
        font-size: 0.8rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Missed-Lead Detector</h1>
    <p>Automated inbox monitoring, smart replies & follow-up management</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")

    # Custom styled radio for navigation
    page = st.radio(
        "Go to",
        ["Dashboard", "Inbox", "Auto-Replies", "Notifications", "Model Performance", "Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick stats in sidebar
    scored_exists = os.path.exists(SCORED)
    notif_count = 0
    if os.path.exists(NOTIF_LOG):
        with open(NOTIF_LOG) as f:
            notifs = json.load(f)
            notif_count = sum(1 for n in notifs if not n.get("read", False))

    if notif_count > 0:
        st.markdown(f"""
        <div style="background: {COLORS['danger_muted']}; border: 1px solid rgba(248, 113, 113, 0.2);
                    border-radius: 8px; padding: 0.75rem 1rem; margin-top: 1rem;">
            <span style="color: {COLORS['danger']}; font-weight: 600; font-size: 0.9rem;">
                {notif_count} unread alert{"s" if notif_count > 1 else ""}
            </span>
        </div>
        """, unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────
def is_live_configured() -> bool:
    return bool(os.getenv("IMAP_USER")) or bool(os.getenv("SMTP_USER"))


@st.cache_data(ttl=30)
def load_scored():
    if os.path.exists(SCORED):
        return pd.read_csv(SCORED)
    return pd.DataFrame()


@st.cache_data(ttl=30)
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return [] if "list" in path else {}


@st.cache_data(ttl=30)
def load_followup_status():
    if os.path.exists(FOLLOWUP_LOG):
        with open(FOLLOWUP_LOG) as f:
            return json.load(f)
    return {}


def render_empty_state(icon: str, title: str, text: str):
    """Render a consistent empty state component."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown('<div class="section-header">Sales Team Overview</div>', unsafe_allow_html=True)

    live = is_live_configured()
    status_text = "Monitoring Gmail" if live else "Using sample data"
    status_class = "status-live" if live else "status-demo"
    st.markdown(f'Pipeline: <span class="status-badge {status_class}">{status_text}</span>',
                unsafe_allow_html=True)

    scored = load_scored()
    followup = load_followup_status()
    reply_log = load_json(REPLY_LOG)

    # Metric cards - Asymmetric layout for visual interest
    col_wide, col_narrow = st.columns([2, 1])

    with col_wide:
        m1, m2 = st.columns(2)
        with m1:
            total = len(scored)
            st.markdown(f"""<div class="metric-card accent-info">
                <div class="metric-value">{total:,}</div>
                <div class="metric-label">Total Leads</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            missed = int(scored["predicted_missed"].sum()) if "predicted_missed" in scored.columns and len(scored) > 0 else 0
            st.markdown(f"""<div class="metric-card accent-danger">
                <div class="metric-value">{missed}</div>
                <div class="metric-label">Missed Leads</div>
            </div>""", unsafe_allow_html=True)

    with col_narrow:
        replied = len([r for r in reply_log if isinstance(r, dict)])
        st.markdown(f"""<div class="metric-card accent-success">
            <div class="metric-value">{replied}</div>
            <div class="metric-label">Auto-Replied</div>
        </div>""", unsafe_allow_html=True)

    # Second row
    m3, m4 = st.columns(2)
    with m3:
        overdue = len([s for s in followup.values()
                      if isinstance(s, dict) and s.get("auto_replied")
                      and not s.get("human_followed_up")])
        st.markdown(f"""<div class="metric-card accent-warning">
            <div class="metric-value">{overdue}</div>
            <div class="metric-label">Awaiting Human</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        if "high_intent_flag" in scored.columns and len(scored) > 0:
            hi = int(scored["high_intent_flag"].sum())
        else:
            hi = 0
        st.markdown(f"""<div class="metric-card accent-gold">
            <div class="metric-value">{hi}</div>
            <div class="metric-label">High Intent</div>
        </div>""", unsafe_allow_html=True)

    # Lead status breakdown
    if len(scored) > 0:
        st.markdown('<div class="section-header">Lead Distribution</div>', unsafe_allow_html=True)

        if "missed_probability" in scored.columns:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor(COLORS["bg_card"])
            ax.set_facecolor(COLORS["bg_card"])

            ax.hist(scored["missed_probability"], bins=20, color=COLORS["danger"],
                   alpha=0.7, edgecolor=COLORS["bg_card"], linewidth=1)
            ax.axvline(0.5, color=COLORS["accent"], linestyle="--", linewidth=2, label="Threshold (0.5)")

            ax.set_xlabel("Missed Probability", color=COLORS["text_secondary"], fontsize=10)
            ax.set_ylabel("Count", color=COLORS["text_secondary"], fontsize=10)
            ax.set_title("Lead Scoring Distribution", color=COLORS["text_primary"], fontsize=12, fontweight=600, pad=10)
            ax.legend(facecolor=COLORS["bg_elevated"], edgecolor=COLORS["border"], labelcolor=COLORS["text_primary"])
            ax.tick_params(colors=COLORS["text_muted"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(COLORS["border"])
            ax.spines["left"].set_color(COLORS["border"])

            plt.tight_layout()
            st.pyplot(fig)

    # Recent inbox emails table
    if len(scored) > 0:
        st.markdown('<div class="section-header">Recent Leads</div>', unsafe_allow_html=True)
        has_gmail = "_customer_name" in scored.columns

        if has_gmail:
            show_cols = [c for c in ["lead_id", "_customer_name", "_customer_email", "_subject",
                                     "response_gap_hrs", "high_intent_flag",
                                     "missed_probability", "predicted_missed"]
                        if c in scored.columns]
            display_df = scored[show_cols].head(20).copy()
            rename = {"lead_id": "ID", "_customer_name": "Customer", "_customer_email": "Email",
                      "_subject": "Subject", "response_gap_hrs": "Gap (hrs)",
                      "high_intent_flag": "Intent", "missed_probability": "Missed %",
                      "predicted_missed": "Status"}
            display_df.columns = [rename.get(c, c) for c in display_df.columns]
        else:
            show_cols = [c for c in ["lead_id", "channel", "message_text",
                                     "response_gap_hrs", "high_intent_flag",
                                     "missed_probability", "predicted_missed"]
                        if c in scored.columns]
            display_df = scored[show_cols].head(20).copy()
            rename = {"lead_id": "ID", "channel": "Channel", "message_text": "Message",
                      "response_gap_hrs": "Gap (hrs)", "high_intent_flag": "Intent",
                      "missed_probability": "Missed %", "predicted_missed": "Status"}
            display_df.columns = [rename.get(c, c) for c in display_df.columns]

        if "Missed %" in display_df.columns:
            display_df["Missed %"] = display_df["Missed %"].apply(lambda x: f"{x:.0%}")
        if "Intent" in display_df.columns:
            display_df["Intent"] = display_df["Intent"].map({1: "HIGH", 0: "low"})
        if "Status" in display_df.columns:
            display_df["Status"] = display_df["Status"].map({1: "Missed", 0: "Responded"})
        if "Message" in display_df.columns:
            display_df["Message"] = display_df["Message"].apply(
                lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else x)

        st.dataframe(display_df, use_container_width=True, height=350)
    else:
        render_empty_state("📭", "No leads yet", "Scan your inbox to start detecting missed leads.")

# ══════════════════════════════════════════════════════════
# PAGE: Inbox
# ══════════════════════════════════════════════════════════
elif page == "Inbox":
    st.markdown('<div class="section-header">Email Inbox</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Scan Inbox Now", type="primary", use_container_width=True):
            with st.spinner("Scanning inbox for new emails..."):
                try:
                    env = os.environ.copy()
                    result = subprocess.run(
                        [sys.executable, "src/inbox_monitor.py"],
                        cwd=os.path.dirname(BASE),
                        capture_output=True, text=True, timeout=120, env=env
                    )
                    st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
                           language="text")
                    if result.stderr:
                        st.warning(result.stderr[-1000:])
                    st.cache_data.clear()
                    st.rerun()
                except subprocess.TimeoutExpired:
                    st.error("Scan timed out.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("Dry Run (no sends)", use_container_width=True):
            with st.spinner("Scanning inbox without sending..."):
                try:
                    env = os.environ.copy()
                    result = subprocess.run(
                        [sys.executable, "src/inbox_monitor.py", "--dry-run"],
                        cwd=os.path.dirname(BASE),
                        capture_output=True, text=True, timeout=120, env=env
                    )
                    st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
                           language="text")
                except Exception as e:
                    st.error(f"Error: {e}")

    scored = load_scored()
    if len(scored) > 0:
        st.markdown(f'<div class="section-header">All Scored Emails ({len(scored)})</div>', unsafe_allow_html=True)

        has_gmail = "_customer_name" in scored.columns
        if has_gmail:
            cols = [c for c in ["lead_id", "_customer_name", "_customer_email", "_subject",
                               "message_text", "response_gap_hrs", "high_intent_flag",
                               "missed_probability", "predicted_missed", "_received_time"]
                   if c in scored.columns]
        else:
            cols = [c for c in ["lead_id", "channel", "message_text", "message_hour",
                               "high_intent_flag", "prev_contacts", "response_gap_hrs",
                               "missed_probability", "predicted_missed"]
                   if c in scored.columns]

        display = scored[cols].copy()
        rename = {"lead_id": "ID", "_customer_name": "Customer", "_customer_email": "Email",
                  "_subject": "Subject", "message_text": "Message", "channel": "Channel",
                  "message_hour": "Hour", "prev_contacts": "Contacts",
                  "response_gap_hrs": "Gap (hrs)", "high_intent_flag": "Intent",
                  "missed_probability": "Missed %", "predicted_missed": "Status",
                  "_received_time": "Received"}
        display.columns = [rename.get(c, c) for c in display.columns]

        if "Missed %" in display.columns:
            display["Missed %"] = display["Missed %"].apply(lambda x: f"{x:.0%}")
        if "Intent" in display.columns:
            display["Intent"] = display["Intent"].map({1: "HIGH", 0: "low"})
        if "Status" in display.columns:
            display["Status"] = display["Status"].map({1: "Missed", 0: "Responded"})
        if "Message" in display.columns:
            display["Message"] = display["Message"].apply(
                lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else x)

        st.dataframe(display, use_container_width=True, height=500)
    else:
        render_empty_state("📭", "No emails scanned yet", "Click 'Scan Inbox Now' to start monitoring your inbox.")

# ══════════════════════════════════════════════════════════
# PAGE: Auto-Replies
# ══════════════════════════════════════════════════════════
elif page == "Auto-Replies":
    st.markdown('<div class="section-header">Auto-Reply Log</div>', unsafe_allow_html=True)
    st.markdown("Every missed lead gets a human-like auto-reply. The client cannot tell it's automated.")

    reply_log = load_json(REPLY_LOG)
    followup = load_followup_status()

    if reply_log:
        reply_df = pd.DataFrame(reply_log)
        st.markdown(f'<div class="section-header">{len(reply_df)} Auto-Replies Sent</div>', unsafe_allow_html=True)

        # Intent breakdown
        if "detected_intent" in reply_df.columns:
            st.markdown("Intent Distribution")
            intent_counts = reply_df["detected_intent"].value_counts()
            st.bar_chart(intent_counts)

        # Reply details
        show_cols = [c for c in ["lead_id", "customer_name", "customer_email",
                                "reply_subject", "detected_intent",
                                "missed_probability", "replied_at"]
                    if c in reply_df.columns]
        display = reply_df[show_cols].copy()
        rename = {"lead_id": "ID", "customer_name": "Customer", "customer_email": "Email",
                  "reply_subject": "Reply Subject", "detected_intent": "Intent",
                  "missed_probability": "Missed %", "replied_at": "Sent At"}
        display.columns = [rename.get(c, c) for c in display.columns]
        if "Missed %" in display.columns:
            display["Missed %"] = display["Missed %"].apply(lambda x: f"{x:.0%}")

        st.dataframe(display, use_container_width=True, height=400)

        # Follow-up tracking
        st.markdown('<div class="section-header">Follow-Up Status</div>', unsafe_allow_html=True)
        if followup:
            fu_data = []
            for lid, status in followup.items():
                if isinstance(status, dict):
                    fu_data.append({
                        "Lead ID": lid,
                        "Customer": status.get("customer_name", ""),
                        "Auto-Replied": "Yes" if status.get("auto_replied") else "No",
                        "Human Follow-Up": "Completed" if status.get("human_followed_up") else "Pending",
                        "Overdue Alert": "Yes" if status.get("overdue_notified") else "—",
                    })
            if fu_data:
                st.dataframe(pd.DataFrame(fu_data), use_container_width=True, height=300)
    else:
        render_empty_state("📭", "No auto-replies sent yet", "Run the inbox monitor to start sending smart replies.")

# ══════════════════════════════════════════════════════════
# PAGE: Notifications
# ══════════════════════════════════════════════════════════
elif page == "Notifications":
    st.markdown('<div class="section-header">Notifications</div>', unsafe_allow_html=True)

    notifs = load_json(NOTIF_LOG)
    unread = [n for n in notifs if not n.get("read", False)]

    if unread:
        col1, col2 = st.columns([3, 1])
        with col1:
            alert_text = "alerts" if len(unread) > 1 else "alert"
            st.markdown(f"**{len(unread)} unread {alert_text}**")
        with col2:
            if st.button("Mark all read", use_container_width=True):
                for n in notifs:
                    n["read"] = True
                with open(NOTIF_LOG, "w") as f:
                    json.dump(notifs, f, indent=2, default=str)
                st.rerun()

        for i, n in enumerate(reversed(notifs)):
            if not n.get("read", False):
                ntype = n.get("type", "info")
                css_class = "auto-reply" if ntype == "auto_reply" else (
                    "overdue" if ntype == "overdue" else "")
                icon = "🆕" if ntype == "new_lead" else (
                    "✅" if ntype == "auto_reply" else (
                    "⚠️" if ntype == "overdue" else "ℹ️"))

                st.markdown(f"""<div class="notif-card {css_class}">
                    <div class="notif-title">{icon} {n.get('title', '')}</div>
                    <div class="notif-time">{n.get('timestamp', '')}</div>
                    <div class="notif-message">{n.get('message', '')}</div>
                </div>""", unsafe_allow_html=True)
    else:
        render_empty_state("✅", "All caught up", "No unread notifications.")

# ══════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════
elif page == "Model Performance":
    data_source = "Merged (13,740 rows)" if os.path.exists(MERGED_DATA) else "Synthetic (500 rows)"
    st.markdown(f'<div class="section-header">Model Performance — {data_source}</div>', unsafe_allow_html=True)

    # Load metrics
    acc, f1, prec, rec = 0.82, 0.59, 0.68, 0.52
    if os.path.exists(REPORT):
        with open(REPORT) as f:
            for line in f.read().split("\n"):
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "accuracy":
                    acc = float(parts[1])
                if len(parts) >= 5 and parts[0] == "Missed":
                    prec, rec, f1 = float(parts[1]), float(parts[2]), float(parts[3])

    # Metrics with consistent styling
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", f"{acc:.0%}")
    with m2:
        st.metric("Missed F1", f"{f1:.3f}")
    with m3:
        st.metric("Missed Precision", f"{prec:.3f}")
    with m4:
        st.metric("Missed Recall", f"{rec:.3f}")

    # Model comparison
    if os.path.exists(MODEL_CMP):
        with open(MODEL_CMP) as f:
            mc = json.load(f)
        if mc.get("models"):
            st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
            rows = [{"Model": n, "AUC": info["auc"]}
                   for n, info in sorted(mc["models"].items(),
                                         key=lambda x: x[1]["auc"], reverse=True)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Charts
    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(CM_IMG):
            from PIL import Image
            st.image(Image.open(CM_IMG), caption="Confusion Matrix", use_container_width=True)
    with col2:
        if os.path.exists(FI_IMG):
            from PIL import Image
            st.image(Image.open(FI_IMG), caption="Feature Importance", use_container_width=True)

    if os.path.exists(REPORT):
        with open(REPORT) as f:
            st.code(f.read(), language="text")

    # Tuning
    if os.path.exists(CMP_CHART):
        st.markdown('<div class="section-header">Before/After Tuning</div>', unsafe_allow_html=True)
        from PIL import Image
        st.image(Image.open(CMP_CHART), use_container_width=True)

    if os.path.exists(XGB_TUNING):
        with open(XGB_TUNING) as f:
            tuning = json.load(f)
        if tuning.get("best_params"):
            st.markdown('<div class="section-header">XGBoost Tuned Hyperparameters</div>', unsafe_allow_html=True)
            st.json(tuning["best_params"])

# ══════════════════════════════════════════════════════════
# PAGE: Settings
# ══════════════════════════════════════════════════════════
elif page == "Settings":
    st.markdown('<div class="section-header">System Settings</div>', unsafe_allow_html=True)

    st.markdown("Connection Status")
    live = is_live_configured()
    if live:
        st.success("Gmail IMAP/SMTP configured")
    else:
        st.warning("Gmail not configured. Set SMTP_USER and SMTP_PASS env vars.")

    st.markdown("How It Works")
    st.markdown("""
    1. **Inbox Monitor** scans Gmail every 5 minutes (via GitHub Actions)
    2. **ML Model** scores each email for missed-lead probability
    3. **Smart Reply Engine** generates a human-like reply based on detected intent
    4. **Auto-Reply** is sent — client cannot tell it's automated
    5. **Notifications** alert the sales team via email + dashboard + desktop
    6. **Follow-Up Tracker** flags leads that need human attention
    """)

    st.markdown("Environment Variables")
    env_vars = {
        "SMTP_USER": os.getenv("SMTP_USER", "—"),
        "SMTP_HOST": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "IMAP_USER": os.getenv("IMAP_USER", "—"),
        "NOTIFY_EMAIL": os.getenv("NOTIFY_EMAIL", "—"),
        "SENDER_NAME": os.getenv("SENDER_NAME", "Sales Team"),
        "COMPANY_NAME": os.getenv("COMPANY_NAME", "Our Company"),
    }
    for k, v in env_vars.items():
        masked = v[:4] + "****" if len(str(v)) > 4 and k.endswith("_PASS") else v
        st.code(f"{k} = {masked}")

# ── Footer ─────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    Missed-Lead Detector · Automated Inbox Monitoring &amp; Follow-Up · Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)
