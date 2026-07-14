"""
dashboard.py — Missed-Lead Detector
Ops Board — split-flap departure-board command interface.

Design language: a dispatch/ops board. Every metric reads like a status
board tile, every list reads like a manifest. Restrained, flat, dark
slate surfaces with a single semantic color system (amber = attention,
rust = risk, teal = resolved) instead of a different color per widget.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import subprocess
import sys
from datetime import datetime, timedelta

# ── Safe Secrets Access ───────────────────────────────────-
# st.secrets.get() raises StreamlitSecretNotFoundError when no
# secrets.toml exists AND no cloud secrets are configured.
# We wrap it so the app never crashes from missing secrets.
def _get_secret(key: str, fallback: str = "") -> str:
    """Safely read from st.secrets with env var and fallback."""
    # Environment variable takes priority (works everywhere)
    env_val = os.getenv(key)
    if env_val:
        return env_val
    # Try st.secrets (may raise if no secrets configured)
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    return fallback

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

sys.path.insert(0, BASE)
import config

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Missed-Lead Ops Board",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  LOGIN GATE — must pass before ANY other UI renders
# ══════════════════════════════════════════════════════════
# Credentials come from Streamlit secrets or env vars.
# Set AUTH_USER / AUTH_PASS in Streamlit Cloud secrets (Settings > Secrets).
_AUTH_USER = _get_secret("AUTH_USER", "admin")
_AUTH_PASS = _get_secret("AUTH_PASS", "admin")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    [data-testid="stSidebar"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .stApp {
        background: #14161b !important;
        display: flex; align-items: center; justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.15, 1])
    with col:
        st.markdown("""
        <div style="
            background: #1c1f26;
            border: 1px solid #2c303a;
            border-top: 2px solid #e8a33d;
            border-radius: 10px;
            padding: 2.4rem 2.4rem 0.5rem;
            margin-top: 8vh;
            font-family: 'Inter', sans-serif;
        ">
            <div style="text-align:center; margin-bottom: 1.6rem;">
                <div style="font-family:'IBM Plex Mono', monospace; font-size:0.68rem; letter-spacing:0.18em;
                            text-transform:uppercase; color:#8b8f99; margin-bottom:0.6rem;">
                    Access Control · Manifest
                </div>
                <h2 style="color:#eae7dd; margin:0; font-weight:700; font-size:1.5rem; font-family:'Space Grotesk', sans-serif;">
                    Missed-Lead Ops Board
                </h2>
                <p style="color:#8b8f99; margin:0.45rem 0 0; font-size:0.85rem;">Sign in to open the board</p>
            </div>
            <div style="border-top: 1px dashed #383c44; margin: 0 -2.4rem 1.6rem;"></div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")

            if submitted:
                if username.strip() == _AUTH_USER and password == _AUTH_PASS:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Try again.")
    st.stop()  # Block everything below until authenticated

# ══════════════════════════════════════════════════════════
#  OPS BOARD DESIGN SYSTEM — CSS
# ══════════════════════════════════════════════════════════
DESIGN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    :root {
        --board-bg: #14161b;
        --board-panel: #1c1f26;
        --board-panel-alt: #21242c;
        --board-line: #2c303a;
        --ink: #eae7dd;
        --ink-dim: #8b8f99;
        --ink-faint: #52565f;
        --flap-amber: #e8a33d;
        --flap-amber-dim: rgba(232, 163, 61, 0.12);
        --flap-rust: #c9503f;
        --flap-rust-dim: rgba(201, 80, 63, 0.12);
        --flap-teal: #4f9c8f;
        --flap-teal-dim: rgba(79, 156, 143, 0.12);
        --radius-sm: 5px;
        --radius-md: 8px;
        --radius-lg: 10px;
    }

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    .stApp {
        background: var(--board-bg) !important;
        background-image: repeating-linear-gradient(
            0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px,
            transparent 1px, transparent 34px
        ) !important;
        color: var(--ink) !important;
    }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--board-line); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }

    @keyframes bd-flap {
        0%   { opacity: 0; transform: perspective(500px) rotateX(-65deg); }
        65%  { opacity: 1; }
        100% { opacity: 1; transform: perspective(500px) rotateX(0deg); }
    }
    @keyframes bd-rise {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes bd-slide-in {
        from { opacity: 0; transform: translateX(16px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes bd-pulse-dot {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.25; }
    }

    .bd-in { animation: bd-rise 0.5s cubic-bezier(0.16, 1, 0.3, 1) both; }
    .bd-in-1 { animation-delay: 0.05s; }
    .bd-in-2 { animation-delay: 0.10s; }
    .bd-in-3 { animation-delay: 0.15s; }
    .bd-in-4 { animation-delay: 0.20s; }
    .bd-in-5 { animation-delay: 0.25s; }

    section[data-testid="stSidebar"] {
        background: var(--board-panel) !important;
        border-right: 1px solid var(--board-line) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: var(--ink) !important;
    }
    section[data-testid="stSidebar"] hr { border-color: var(--board-line) !important; }

    /* ── Header strip ─────────────────────────────────── */
    .bd-header {
        position: relative;
        background: var(--board-panel);
        border: 1px solid var(--board-line);
        border-top: 2px solid var(--flap-amber);
        border-radius: var(--radius-lg);
        padding: 1.5rem 1.9rem;
        margin-bottom: 1.6rem;
    }
    .bd-header .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin: 0 0 0.35rem;
    }
    .bd-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.65rem;
        letter-spacing: -0.01em;
        color: var(--ink);
        margin: 0;
    }
    .bd-header p {
        color: var(--ink-dim);
        margin-top: 0.4rem;
        font-size: 0.9rem;
        margin-bottom: 0;
    }

    /* ── Generic panel/card ───────────────────────────── */
    .bd-card {
        position: relative;
        background: var(--board-panel);
        border: 1px solid var(--board-line);
        border-radius: var(--radius-md);
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.4rem;
        transition: border-color 0.25s ease, transform 0.25s ease;
    }
    .bd-card:hover { border-color: var(--ink-faint); }
    .bd-card h4, .bd-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.02rem;
        color: var(--ink);
        margin: 0 0 1rem;
    }

    /* ── Split-flap KPI tiles ──────────────────────────── */
    .bd-tile-row { display: flex; gap: 0.9rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
    .bd-tile {
        position: relative;
        flex: 1 1 0;
        min-width: 140px;
        background: var(--board-panel);
        border: 1px solid var(--board-line);
        border-left: 3px solid var(--ink-faint);
        border-radius: var(--radius-md);
        padding: 1.05rem 1.2rem 0.95rem;
        overflow: hidden;
        transform-origin: top center;
        animation: bd-flap 0.55s cubic-bezier(0.2, 0.8, 0.2, 1) both;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .bd-tile:hover { transform: translateY(-2px); }
    .bd-tile::after {
        content: '';
        position: absolute;
        left: 0; right: 0; top: 50%;
        height: 1px;
        background: var(--board-bg);
        opacity: 0.6;
    }
    .bd-tile-value {
        font-family: 'IBM Plex Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-size: 2.05rem;
        font-weight: 600;
        line-height: 1;
        letter-spacing: -0.02em;
        color: var(--ink);
    }
    .bd-tile-label {
        font-size: 0.68rem;
        color: var(--ink-dim);
        margin-top: 0.55rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-family: 'IBM Plex Mono', monospace;
    }
    .bd-tile-label.flag { color: var(--flap-amber); }
    .bd-tile.rust  { border-left-color: var(--flap-rust); }
    .bd-tile.rust  .bd-tile-value { color: var(--flap-rust); }
    .bd-tile.teal  { border-left-color: var(--flap-teal); }
    .bd-tile.teal  .bd-tile-value { color: var(--flap-teal); }
    .bd-tile.amber { border-left-color: var(--flap-amber); }
    .bd-tile.amber .bd-tile-value { color: var(--flap-amber); }

    /* ── Status tags / badges ─────────────────────────── */
    .bd-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.8rem;
        border-radius: var(--radius-sm);
        font-size: 0.66rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .bd-tag.live {
        background: var(--flap-teal-dim);
        color: var(--flap-teal);
        border: 1px solid rgba(79, 156, 143, 0.3);
    }
    .bd-tag.live::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%;
        background: var(--flap-teal); animation: bd-pulse-dot 1.8s infinite;
    }
    .bd-tag.offline {
        background: rgba(139, 143, 153, 0.08);
        color: var(--ink-dim);
        border: 1px solid var(--board-line);
    }
    .bd-tag.offline::before {
        content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--ink-faint);
    }

    /* ── Buttons ───────────────────────────────────────── */
    .stButton > button {
        background: var(--board-panel-alt) !important;
        color: var(--ink) !important;
        border: 1px solid var(--board-line) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1.4rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em !important;
        transition: border-color 0.2s ease, transform 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--flap-amber) !important;
        color: var(--ink) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: var(--flap-amber) !important;
        color: #1a1408 !important;
        border: 1px solid var(--flap-amber) !important;
        font-weight: 600 !important;
    }
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #f2b055 !important;
        border-color: #f2b055 !important;
    }

    /* ── Inputs ────────────────────────────────────────── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    textarea {
        background: var(--board-panel-alt) !important;
        border: 1px solid var(--board-line) !important;
        color: var(--ink) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] input:hover,
    textarea:hover { border-color: var(--ink-faint) !important; }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] input:focus,
    textarea:focus {
        border-color: var(--flap-amber) !important;
        box-shadow: 0 0 0 3px var(--flap-amber-dim) !important;
    }

    /* ── Notification / ledger rows ───────────────────── */
    .bd-notif {
        background: var(--board-panel-alt);
        border: 1px solid var(--board-line);
        border-left: 3px solid var(--ink-faint);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1rem;
        margin-bottom: 0.55rem;
        animation: bd-slide-in 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .bd-notif.new_lead   { border-left-color: var(--flap-amber); }
    .bd-notif.auto_reply { border-left-color: var(--flap-teal); }
    .bd-notif.overdue    { border-left-color: var(--flap-rust); }
    .bd-notif.info       { border-left-color: var(--ink-faint); }
    .bd-notif-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-right: 0.5rem;
    }
    .bd-notif.new_lead   .bd-notif-tag { color: var(--flap-amber); }
    .bd-notif.auto_reply .bd-notif-tag { color: var(--flap-teal); }
    .bd-notif.overdue    .bd-notif-tag { color: var(--flap-rust); }
    .bd-notif.info       .bd-notif-tag { color: var(--ink-faint); }

    /* ── Terminal / log output ────────────────────────── */
    .bd-terminal {
        background: #0e1013 !important;
        border: 1px solid var(--board-line) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.1rem !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.76rem !important;
        color: var(--flap-amber) !important;
        line-height: 1.6 !important;
        max-height: 400px;
        overflow-y: auto;
    }

    /* ── Reply preview card ───────────────────────────── */
    .bd-email {
        background: var(--board-panel-alt);
        border: 1px solid var(--board-line);
        border-radius: var(--radius-md);
        padding: 1.2rem;
        margin-top: 0.8rem;
    }
    .bd-email-header {
        border-bottom: 1px dashed var(--board-line);
        padding-bottom: 0.7rem;
        margin-bottom: 0.7rem;
        font-size: 0.78rem;
        color: var(--ink-dim);
        font-family: 'IBM Plex Mono', monospace;
    }
    .bd-email-header span { color: var(--ink); font-weight: 500; }
    .bd-email-body {
        font-size: 0.88rem;
        color: var(--ink);
        line-height: 1.6;
        white-space: pre-wrap;
        font-family: 'Inter', sans-serif;
    }

    /* ── Tabs ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        background: var(--board-panel-alt);
        border-radius: var(--radius-sm);
        padding: 0.25rem;
        border: 1px solid var(--board-line);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 4px;
        color: var(--ink-dim) !important;
        font-weight: 500;
        padding: 0.35rem 1.1rem;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--flap-amber-dim) !important;
        color: var(--flap-amber) !important;
        border: 1px solid rgba(232, 163, 61, 0.25) !important;
    }

    [data-testid="stDialog"] {
        background: var(--board-panel) !important;
        border: 1px solid var(--board-line) !important;
        border-radius: var(--radius-lg) !important;
    }

    .bd-sidebar-box {
        background: var(--board-panel-alt);
        border: 1px solid var(--board-line);
        border-radius: var(--radius-sm);
        padding: 0.75rem 0.95rem;
        font-size: 0.79rem;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.85;
        color: var(--ink-dim);
    }
    .bd-sidebar-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin: 0 0 0.6rem;
    }

    #MainMenu, footer { visibility: hidden; }

    .bd-footer {
        text-align: center;
        padding: 2.2rem 0 1.4rem 0;
        color: var(--ink-faint);
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        border-top: 1px solid var(--board-line);
        margin-top: 3rem;
        letter-spacing: 0.05em;
    }

    [data-testid="stMetric"] {
        background: var(--board-panel-alt);
        border: 1px solid var(--board-line);
        border-radius: var(--radius-sm);
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stMetricLabel"] { color: var(--ink-dim) !important; }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--board-line);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    @media (max-width: 768px) {
        .bd-header { padding: 1.2rem 1.1rem; }
        .bd-header h1 { font-size: 1.3rem; }
        .bd-header p { font-size: 0.82rem; }
        .bd-tile-row { flex-wrap: wrap; }
        .bd-tile { flex: 1 1 calc(50% - 0.45rem); min-width: calc(50% - 0.45rem); padding: 0.9rem 1rem; }
        .bd-tile-value { font-size: 1.5rem; }
        .bd-card { padding: 1.1rem; }
    }
    @media (max-width: 480px) {
        .bd-tile { flex: 1 1 100%; min-width: 100%; }
        .bd-header h1 { font-size: 1.2rem; }
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
    return {} if "followup" in path.lower() or "config" in path.lower() else []

# ── Connection Checks ──────────────────────────────────────
def is_live_mode() -> bool:
    return bool(os.getenv("IMAP_USER")) or bool(os.getenv("SMTP_USER"))

# ══════════════════════════════════════════════════════════
#  LEAD DETAIL DIALOG (Modal)
# ══════════════════════════════════════════════════════════
@st.dialog("Lead Details", width="large")
def show_lead_dialog(lead_data, has_gmail_headers):
    """Modal dialog showing full lead details and smart reply preview."""
    status_tag = "MISSED" if lead_data.get('predicted_missed') == 1 else "RESPONDED"
    st.markdown(f"""
    <div style="margin-bottom: 1.4rem;">
        <div class="bd-tag {'offline' if lead_data.get('predicted_missed') == 1 else 'live'}" style="margin-bottom:0.6rem;">{status_tag}</div>
        <h2 style="margin:0; color: var(--ink); font-weight: 700; font-family:'Space Grotesk',sans-serif;">
            Lead {lead_data.get('lead_id', 'N/A')}
        </h2>
        <p style="color: var(--ink-dim); margin-top: 0.25rem; font-size: 0.88rem;">
            Full lead record and smart reply preview
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Lead information")
        info_items = [
            ("Lead ID", f"`{lead_data.get('lead_id', 'N/A')}`"),
        ]
        if has_gmail_headers:
            info_items.extend([
                ("Customer", f"{lead_data.get('_customer_name', 'N/A')}"),
                ("Email", f"`{lead_data.get('_customer_email', 'N/A')}`"),
                ("Subject", f"{lead_data.get('_subject', 'N/A')}"),
                ("Received", f"{lead_data.get('_received_time', 'Unknown')}"),
            ])
        info_items.extend([
            ("Channel", f"{lead_data.get('channel', 'N/A')}"),
            ("Response gap", f"**{lead_data.get('response_gap_hrs', 0):.1f}** hours"),
            ("Risk score", f"**{lead_data.get('missed_probability', 0):.1%}**"),
            ("Status", status_tag),
        ])
        for label, value in info_items:
            st.markdown(f"**{label}:** {value}")

        st.markdown("---")
        st.markdown("#### Original message")
        st.info(lead_data.get('message_text', 'No message content available.'))

    with c2:
        st.markdown("#### Smart auto-reply preview")
        try:
            from smart_reply_engine import generate_reply
            reply_payload = generate_reply(
                customer_name=lead_data.get("_customer_name") or "Valued Customer",
                customer_email=lead_data.get("_customer_email") or "customer@example.com",
                subject=lead_data.get("_subject") or "Enquiry",
                message_text=lead_data.get("message_text", ""),
                channel=lead_data.get("channel", "Email"),
            )
            st.markdown(f"""
            <span class="bd-tag amber-tag" style="background: var(--flap-amber-dim); color: var(--flap-amber); border: 1px solid rgba(232,163,61,0.25); margin-bottom: 0.8rem;">
                Intent: {reply_payload['detected_intent'].upper()}
            </span>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="bd-email">
                <div class="bd-email-header">
                    <div>From: <span>Sales Team &lt;noreply@yourcompany.com&gt;</span></div>
                    <div>To: <span>{lead_data.get('_customer_name', 'Valued Customer')} &lt;{lead_data.get('_customer_email', 'customer@example.com')}&gt;</span></div>
                    <div>Subject: <span>{reply_payload["reply_subject"]}</span></div>
                </div>
                <div class="bd-email-body">{reply_payload["reply_body"]}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not load smart reply templates: {e}")


# ── Header Section ─────────────────────────────────────────
live_connected = is_live_mode()
status_label = "Live" if live_connected else "Not connected"
status_class = "live" if live_connected else "offline"

st.markdown(f"""
<div class="bd-header bd-in">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
        <div>
            <p class="eyebrow">Ops Board · Inbox Monitoring</p>
            <h1>Missed-Lead Ops Board</h1>
        </div>
        <span class="bd-tag {status_class}">{status_label}</span>
    </div>
    <p>Tracks inbound leads, flags the ones at risk of going unanswered, and fires smart auto-replies.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configurations & Navigation ───────────────────
with st.sidebar:
    st.markdown('<p class="bd-sidebar-eyebrow">Navigation</p>', unsafe_allow_html=True)
    page = st.radio(
        "Go to:",
        ["Command Center", "Lead Explorer", "Auto-Replies Tracker",
         "Interactive Pipeline Graph", "Performance Overview", "Workflow Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p class="bd-sidebar-eyebrow">System status</p>', unsafe_allow_html=True)

    leads_df = load_scored_leads()
    notifs = load_json_log(NOTIF_LOG)
    unread_notifs = [n for n in notifs if not n.get("read", False)] if isinstance(notifs, list) else []

    st.markdown(f"""
    <div class="bd-sidebar-box">
        <div>Gmail &nbsp;&nbsp;{'CONNECTED' if live_connected else 'OFFLINE'}</div>
        <div>Scored &nbsp;&nbsp;{len(leads_df)} leads</div>
        <div>Alerts &nbsp;&nbsp;<span style="color: {'var(--flap-rust)' if len(unread_notifs) > 0 else 'var(--ink-dim)'}; font-weight: bold;">{len(unread_notifs)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")



# ══════════════════════════════════════════════════════════
#  PAGE: Command Center
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

    # ── Empty state ──────────────────────────────────────
    if total_leads == 0:
        st.markdown(f"""
        <div class="bd-card" style="text-align:center; padding: 2.6rem 2rem;">
            <p class="bd-sidebar-eyebrow" style="text-align:center;">No inbox data yet</p>
            <h2 style="color: var(--ink); margin: 0.3rem 0 0.6rem; font-weight: 700; font-family:'Space Grotesk',sans-serif;">Connect Gmail to start scanning</h2>
            <p style="color: var(--ink-dim); font-size: 0.92rem; max-width: 520px; margin: 0 auto;">
                Once connected, the board fills in with real customer emails and risk scores.
            </p>
            <div style="margin-top: 1.4rem; padding: 1.2rem; background: var(--board-panel-alt); border-radius: var(--radius-md); border: 1px solid var(--board-line); max-width: 550px; margin-left: auto; margin-right: auto; text-align: left;">
                <p style="color: var(--flap-amber); font-weight: 600; margin-bottom: 0.65rem; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em;">Setup steps</p>
                <p style="color: var(--ink-dim); font-size: 0.85rem; line-height: 2;">
                    1. Enable 2FA on your Gmail and generate an App Password<br>
                    2. Add these to Streamlit Cloud secrets:<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--flap-amber);">IMAP_USER</code> = your Gmail<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--flap-amber);">IMAP_PASS</code> = App Password<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--flap-amber);">SMTP_USER</code> = your Gmail<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--flap-amber);">SMTP_PASS</code> = App Password<br>
                    3. Select "Run inbox scan" below to fetch emails
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────
    st.markdown(f"""
    <div class="bd-tile-row">
        <div class="bd-tile bd-in-1">
            <div class="bd-tile-value">{total_leads:,}</div>
            <div class="bd-tile-label">Scanned leads</div>
        </div>
        <div class="bd-tile rust bd-in-2">
            <div class="bd-tile-value">{missed_leads}</div>
            <div class="bd-tile-label">Missed leads</div>
        </div>
        <div class="bd-tile teal bd-in-3">
            <div class="bd-tile-value">{replied_count}</div>
            <div class="bd-tile-label">Auto-replied</div>
        </div>
        <div class="bd-tile amber bd-in-4">
            <div class="bd-tile-value">{overdue_count}</div>
            <div class="bd-tile-label">Awaiting action</div>
        </div>
        <div class="bd-tile bd-in-5">
            <div class="bd-tile-value">{high_intent}</div>
            <div class="bd-tile-label flag">High intent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Layout ──────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Pipeline Controls
        st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
        st.markdown("<p class='bd-card-title'>Inbox monitoring pipeline</p>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            scan_btn = st.button("Run inbox scan", type="primary", use_container_width=True)
        with c2:
            dry_btn = st.button("Dry run (no send)", use_container_width=True)

        if scan_btn or dry_btn:
            is_dry = bool(dry_btn)
            st.markdown("**Pipeline output:**")
            with st.spinner("Scanning Gmail inbox..."):
                try:
                    cmd = [sys.executable, os.path.join(BASE, "inbox_monitor.py")]
                    if is_dry:
                        cmd.append("--dry-run")

                    # Pass credentials into the subprocess environment so
                    # email_reader.py / inbox_monitor.py can connect to Gmail.
                    # Priority: environment variable already set > .bat config fallback.
                    scan_env = os.environ.copy()
                    scan_env.setdefault("IMAP_USER",  _get_secret("IMAP_USER"))
                    scan_env.setdefault("IMAP_PASS",  _get_secret("IMAP_PASS"))
                    scan_env.setdefault("SMTP_USER",  _get_secret("SMTP_USER"))
                    scan_env.setdefault("SMTP_PASS",  _get_secret("SMTP_PASS"))
                    scan_env.setdefault("SENDER_NAME", os.getenv("SENDER_NAME", "Sales Team"))
                    scan_env["STREAMLIT_SERVER_HEADLESS"] = "true"

                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120,   # Gmail fetch can take a bit
                        env=scan_env,
                    )
                    stdout = res.stdout if res.stdout else "No output returned."
                    st.markdown(f"<pre class='bd-terminal'>{stdout}</pre>", unsafe_allow_html=True)
                    if res.stderr:
                        with st.expander("Warnings / stderr"):
                            st.code(res.stderr)
                    st.cache_data.clear()
                    st.rerun()
                except subprocess.TimeoutExpired:
                    st.error("Scan timed out after 120 seconds. Gmail may be slow — try again.")
                except Exception as e:
                    st.error(f"Execution error: {e}")
        elif not live_connected:
            st.warning("Gmail not connected. Set IMAP_USER and IMAP_PASS in Streamlit secrets.")
        else:
            st.markdown("<p style='color: var(--ink-dim); font-style: italic;'>Run a scan to read Gmail, score leads, and fire automatic replies.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Distribution Chart
        st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
        st.markdown("<p class='bd-card-title'>Lead risk distribution</p>", unsafe_allow_html=True)

        if total_leads > 0 and "missed_probability" in scored.columns:
            if HAS_PLOTLY:
                fig = px.histogram(
                    scored, x="missed_probability", nbins=20,
                    title="Lead risk distribution (threshold = 0.50)",
                    labels={"missed_probability": "Predicted missed probability", "count": "Lead count"},
                    color_discrete_sequence=["#e8a33d"],
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b8f99',
                    font_family='IBM Plex Mono, monospace',
                    title_font_color='#eae7dd',
                    showlegend=False,
                    xaxis=dict(showgrid=False, linecolor='#2c303a'),
                    yaxis=dict(showgrid=True, gridcolor='#2c303a', linecolor='#2c303a'),
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="#c9503f",
                              annotation_text="Threshold", annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)
            else:
                if os.path.exists(CM_IMG) and HAS_PIL:
                    st.image(Image.open(CM_IMG), use_container_width=True)
        else:
            st.info("No lead metrics scored yet. Run a scan above to ingest data.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='bd-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("<p class='bd-card-title'>Notification feed</p>", unsafe_allow_html=True)

        if isinstance(notifs, list) and notifs:
            if len(unread_notifs) > 0:
                if st.button(f"Mark all read ({len(unread_notifs)})", use_container_width=True):
                    for n in notifs:
                        n["read"] = True
                    with open(NOTIF_LOG, "w") as f:
                        json.dump(notifs, f, indent=2, default=str)
                    st.rerun()

            unread_count = 0
            tag_names = {"new_lead": "NEW", "auto_reply": "AUTO", "overdue": "LATE"}
            for i, n in enumerate(reversed(notifs)):
                if not n.get("read", False) and unread_count < 10:
                    ntype = n.get("type", "info")
                    tag = tag_names.get(ntype, "INFO")

                    st.markdown(f"""
                    <div class="bd-notif {ntype}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2rem;">
                            <span style="font-weight: 600; font-size: 0.86rem; color: var(--ink);"><span class="bd-notif-tag">{tag}</span>{n.get('title', 'Notification')}</span>
                            <span style="font-size: 0.68rem; color: var(--ink-faint); font-family: 'IBM Plex Mono', monospace;">{n.get('timestamp', '')[11:16]}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--ink-dim); line-height: 1.4; margin-top: 0.2rem;">{n.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    unread_count += 1
            if unread_count == 0:
                st.markdown("<p style='text-align:center; color:var(--ink-faint); padding: 2rem 0;'>All caught up.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center; color:var(--ink-faint); padding: 2rem 0;'>No notifications yet.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: Lead Explorer
# ══════════════════════════════════════════════════════════
elif page == "Lead Explorer":
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Lead table &amp; inspector</p>", unsafe_allow_html=True)

    scored = load_scored_leads()

    if scored.empty:
        st.warning("No leads recorded. Run an inbox scan to fetch records.")
    else:
        has_gmail_headers = "_customer_name" in scored.columns

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            q_search = st.text_input("Search customer / email", "")
        with c2:
            q_status = st.selectbox("Pipeline status", ["All", "Missed leads", "Responded (low risk)"])
        with c3:
            q_intent = st.selectbox("High intent", ["All", "Yes", "No"])
        with c4:
            channels = list(scored["channel"].unique()) if "channel" in scored.columns else ["Gmail"]
            q_channel = st.selectbox("Channel", ["All"] + channels)

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
            val = 1 if q_status == "Missed leads" else 0
            filtered = filtered[filtered["predicted_missed"] == val]
        if q_intent != "All":
            val = 1 if q_intent == "Yes" else 0
            filtered = filtered[filtered["high_intent_flag"] == val]
        if q_channel != "All" and "channel" in filtered.columns:
            filtered = filtered[filtered["channel"] == q_channel]

        if has_gmail_headers:
            show_cols = ["lead_id", "_customer_name", "_customer_email", "_subject", "channel",
                         "response_gap_hrs", "high_intent_flag", "missed_probability", "predicted_missed"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            rename_map = {
                "lead_id": "Lead ID", "_customer_name": "Customer",
                "_customer_email": "Email", "_subject": "Subject",
                "channel": "Source", "response_gap_hrs": "Gap (hrs)",
                "high_intent_flag": "High intent", "missed_probability": "Risk score",
                "predicted_missed": "Missed",
            }
        else:
            show_cols = ["lead_id", "channel", "message_text", "response_gap_hrs",
                         "high_intent_flag", "missed_probability", "predicted_missed"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            rename_map = {
                "lead_id": "Lead ID", "channel": "Source",
                "message_text": "Inquiry", "response_gap_hrs": "Gap (hrs)",
                "high_intent_flag": "High intent", "missed_probability": "Risk score",
                "predicted_missed": "Missed",
            }

        display_df = filtered[show_cols].copy()
        if "missed_probability" in display_df.columns:
            display_df["missed_probability"] = display_df["missed_probability"].apply(lambda x: f"{x:.1%}")
        if "predicted_missed" in display_df.columns:
            display_df["predicted_missed"] = display_df["predicted_missed"].map({1: "MISSED", 0: "Responded"})
        if "high_intent_flag" in display_df.columns:
            display_df["high_intent_flag"] = display_df["high_intent_flag"].map({1: "High", 0: "Normal"})
        display_df = display_df.rename(columns=rename_map)

        st.dataframe(display_df, use_container_width=True, height=300)

        st.markdown("---")
        st.markdown("<p class='bd-card-title'>Lead inspector</p>", unsafe_allow_html=True)

        selected_id = st.selectbox("Select a lead ID:", ["-- None --"] + list(filtered["lead_id"].unique()))

        if selected_id != "-- None --":
            lead_row = filtered[filtered["lead_id"] == selected_id].iloc[0].to_dict()

            d_col1, d_col2 = st.columns([1, 1])

            with d_col1:
                st.markdown("<div style='background:var(--board-panel-alt); padding:1.2rem; border-radius:var(--radius-sm); border:1px solid var(--board-line);'>", unsafe_allow_html=True)
                st.markdown(f"**Lead ID:** `{lead_row.get('lead_id', 'N/A')}`")

                if has_gmail_headers:
                    st.markdown(f"**Customer:** {lead_row.get('_customer_name', 'N/A')}")
                    st.markdown(f"**Email:** `{lead_row.get('_customer_email', 'N/A')}`")
                    st.markdown(f"**Subject:** {lead_row.get('_subject', 'N/A')}")
                    st.markdown(f"**Received:** {lead_row.get('_received_time', 'Unknown')}")
                else:
                    st.markdown(f"**Source:** {lead_row.get('channel', 'N/A')}")

                st.markdown(f"**Response gap:** {lead_row.get('response_gap_hrs', 0):.1f} hours")
                st.markdown(f"**Risk score:** `{lead_row.get('missed_probability', 0):.2f}`")
                st.markdown(f"**Status:** {'MISSED LEAD' if lead_row.get('predicted_missed') == 1 else 'Responded'}")

                st.markdown("**Original message:**")
                st.info(lead_row.get("message_text", "No message"))
                st.markdown("</div>", unsafe_allow_html=True)

            with d_col2:
                st.markdown("**Smart auto-reply preview:**")
                try:
                    from smart_reply_engine import generate_reply
                    reply_payload = generate_reply(
                        customer_name=lead_row.get("_customer_name") or "Valued Customer",
                        customer_email=lead_row.get("_customer_email") or "customer@example.com",
                        subject=lead_row.get("_subject") or "Enquiry",
                        message_text=lead_row.get("message_text", ""),
                        channel=lead_row.get("channel", "Email"),
                    )
                    st.markdown(f"""
                    <span class="bd-tag" style="background: var(--flap-amber-dim); color: var(--flap-amber); border: 1px solid rgba(232,163,61,0.25); margin-bottom: 0.8rem;">
                        Intent: {reply_payload['detected_intent'].upper()}
                    </span>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="bd-email">
                        <div class="bd-email-header">
                            <div>From: <span>Sales Team &lt;noreply@yourcompany.com&gt;</span></div>
                            <div>To: <span>{lead_row.get('_customer_name', 'Valued Customer')} &lt;{lead_row.get('_customer_email', 'customer@example.com')}&gt;</span></div>
                            <div>Subject: <span>{reply_payload["reply_subject"]}</span></div>
                        </div>
                        <div class="bd-email-body">{reply_payload["reply_body"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not load smart reply templates: {e}")

            # Open in Modal button
            if st.button("View full detail", use_container_width=True):
                show_lead_dialog(lead_row, has_gmail_headers)
        else:
            st.info("Pick a lead ID above to drill down into details.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: Auto-Replies Tracker
# ══════════════════════════════════════════════════════════
elif page == "Auto-Replies Tracker":
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Automated auto-replies history</p>", unsafe_allow_html=True)

    reply_log = load_json_log(REPLY_LOG)
    followup_status = load_json_log(FOLLOWUP_LOG)

    if not reply_log:
        st.info("No auto-replies logged yet. They're generated when the inbox scan detects new missed leads.")
    else:
        reply_df = pd.DataFrame(reply_log)
        st.markdown("---")
        st.markdown("<p class='bd-card-title'>Sent reply logs</p>", unsafe_allow_html=True)
        show_cols = [c for c in ["lead_id", "customer_name", "customer_email", "reply_subject", "detected_intent", "replied_at"] if c in reply_df.columns]
        st.dataframe(reply_df[show_cols], use_container_width=True)

        st.markdown("---")
        st.markdown("<p class='bd-card-title'>Overdue human follow-up</p>", unsafe_allow_html=True)

        if followup_status:
            fu_items = []
            for lid, info in followup_status.items():
                if isinstance(info, dict):
                    fu_items.append({
                        "lead_id": lid,
                        "customer_name": info.get("customer_name", ""),
                        "customer_email": info.get("customer_email", ""),
                        "auto_replied_at": info.get("auto_replied_at", "—"),
                        "human_follow_up": "Done" if info.get("human_followed_up") else "Pending",
                        "alert_escalated": "Overdue" if info.get("overdue_notified") else "—",
                    })

            fu_df = pd.DataFrame(fu_items)
            pending_leads = [item["lead_id"] for item in fu_items if "Pending" in item["human_follow_up"]]

            if pending_leads:
                col_sel, col_act = st.columns([2, 1])
                with col_sel:
                    action_id = st.selectbox("Mark lead as resolved", ["-- Select --"] + pending_leads)
                with col_act:
                    st.markdown("<div style='margin-top:1.75rem;'></div>", unsafe_allow_html=True)
                    if st.button("Mark resolved", use_container_width=True) and action_id != "-- Select --":
                        followup_status[action_id]["human_followed_up"] = True
                        followup_status[action_id]["human_followed_up_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(FOLLOWUP_LOG, "w") as f:
                            json.dump(followup_status, f, indent=2, default=str)
                        st.success(f"Lead {action_id} marked as resolved.")
                        st.rerun()

            st.dataframe(fu_df, use_container_width=True)
        else:
            st.info("No leads awaiting manual follow-up.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: Interactive Pipeline Graph
# ══════════════════════════════════════════════════════════
elif page == "Interactive Pipeline Graph":
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Lead flow pipeline</p>", unsafe_allow_html=True)

    scored = load_scored_leads()
    reply_log = load_json_log(REPLY_LOG)

    if len(scored) == 0:
        st.info("No leads available to visualize.")
    else:
        st.markdown("<p style='color: var(--ink-dim);'>How leads move through the pipeline, from first contact to resolution.</p>", unsafe_allow_html=True)
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
                    line=dict(color="#2c303a", width=0.5),
                    label=["Total leads", "Responded (safe)", "Missed leads",
                           "High intent (missed)", "Low intent (missed)",
                           "Auto-replied", "Awaiting human"],
                    color=["#8b8f99", "#4f9c8f", "#c9503f", "#e8a33d", "#52565f", "#4f9c8f", "#c9503f"],
                ),
                link=dict(
                    source=[0, 0, 2, 2, 3, 3],
                    target=[1, 2, 3, 4, 5, 6],
                    value=[max(responded, 1), max(missed, 1), max(high_intent_missed, 1),
                           max(low_intent_missed, 1), max(auto_replied, 1), max(awaiting, 1)],
                    color=["rgba(79,156,143,0.2)", "rgba(201,80,63,0.2)", "rgba(232,163,61,0.2)",
                           "rgba(82,86,95,0.2)", "rgba(79,156,143,0.2)", "rgba(201,80,63,0.2)"],
                ),
            )])
            fig.update_layout(
                title_text="Customer journey &amp; sales bottlenecks",
                font_size=13, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='#eae7dd',
                font_family='IBM Plex Mono, monospace', height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Plotly is required to render the interactive graph.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: Performance Overview
# ══════════════════════════════════════════════════════════
elif page == "Performance Overview":
    from datetime import timedelta

    scored = load_scored_leads()
    reply_log = load_json_log(REPLY_LOG)
    followup_status = load_json_log(FOLLOWUP_LOG)

    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Performance overview</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--ink-dim); margin-top:-0.5rem;'>Your sales lead recovery at a glance.</p>", unsafe_allow_html=True)

    now = datetime.now()
    week_ago = now - timedelta(days=7)
    two_weeks_ago = now - timedelta(days=14)

    total_leads = len(scored)
    missed_total = 0
    missed_this_week = 0
    missed_last_week = 0
    responded_count = 0

    if total_leads > 0 and "predicted_missed" in scored.columns:
        missed_total = int(scored["predicted_missed"].sum())
        responded_count = total_leads - missed_total

        if "_received_time" in scored.columns:
            try:
                scored["_parsed_time"] = pd.to_datetime(scored["_received_time"], format="%Y-%m-%d %H:%M UTC", errors="coerce")
                this_week_mask = scored["_parsed_time"] >= pd.Timestamp(week_ago)
                last_week_mask = (scored["_parsed_time"] >= pd.Timestamp(two_weeks_ago)) & (scored["_parsed_time"] < pd.Timestamp(week_ago))
                missed_this_week = int(scored.loc[this_week_mask & (scored["predicted_missed"] == 1)].shape[0])
                missed_last_week = int(scored.loc[last_week_mask & (scored["predicted_missed"] == 1)].shape[0])
            except Exception:
                pass

    auto_replied_leads = set()
    if isinstance(reply_log, list):
        for r in reply_log:
            if isinstance(r, dict) and r.get("lead_id"):
                auto_replied_leads.add(r["lead_id"])
    auto_reply_count = len(auto_replied_leads)

    recovered = 0
    if isinstance(followup_status, dict):
        recovered = sum(1 for s in followup_status.values() if isinstance(s, dict) and s.get("human_followed_up"))

    handled_leads = set(auto_replied_leads)
    if isinstance(followup_status, dict):
        for lid, s in followup_status.items():
            if isinstance(s, dict) and (s.get("auto_replied") or s.get("human_followed_up")):
                handled_leads.add(lid)
    recovery_rate = (len(handled_leads) / missed_total * 100) if missed_total > 0 else 0
    leads_saved = len(handled_leads)

    # KPI Cards
    st.markdown(f"""
    <div class="bd-tile-row">
        <div class="bd-tile rust bd-in-1">
            <div class="bd-tile-value">{missed_total}</div>
            <div class="bd-tile-label">Missed detected</div>
        </div>
        <div class="bd-tile teal bd-in-2">
            <div class="bd-tile-value">{auto_reply_count}</div>
            <div class="bd-tile-label">Auto follow-ups</div>
        </div>
        <div class="bd-tile bd-in-3">
            <div class="bd-tile-value">{recovery_rate:.0f}%</div>
            <div class="bd-tile-label">Recovery rate</div>
        </div>
        <div class="bd-tile teal bd-in-4">
            <div class="bd-tile-value">{leads_saved}</div>
            <div class="bd-tile-label">Leads saved</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # This Week vs Last Week
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>This week vs last week</p>", unsafe_allow_html=True)

    week_col1, week_col2 = st.columns(2)
    with week_col1:
        delta_w = missed_this_week - missed_last_week
        delta_color = "normal" if delta_w <= 0 else "inverse"
        st.metric(label="Missed this week", value=missed_this_week,
                  delta=f"{delta_w:+d} vs last week" if missed_last_week > 0 else None,
                  delta_color=delta_color)
    with week_col2:
        st.metric(label="Missed last week", value=missed_last_week)

    if HAS_PLOTLY and (missed_this_week > 0 or missed_last_week > 0):
        fig = go.Figure(data=[
            go.Bar(name='Last week', x=['Missed leads'], y=[missed_last_week],
                   marker_color='#52565f'),
            go.Bar(name='This week', x=['Missed leads'], y=[missed_this_week],
                   marker_color='#c9503f'),
        ])
        fig.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#8b8f99', font_family='IBM Plex Mono, monospace', showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=20, b=40),
            yaxis=dict(showgrid=True, gridcolor='#2c303a'),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 4-Week Trend
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>4-week trend</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--ink-dim); margin-top:-0.5rem;'>Missed leads over the past 4 weeks to spot patterns.</p>", unsafe_allow_html=True)

    if total_leads > 0 and "_parsed_time" in scored.columns and HAS_PLOTLY:
        try:
            four_weeks_ago = now - timedelta(days=28)
            trend_df = scored[scored["_parsed_time"] >= pd.Timestamp(four_weeks_ago)].copy()

            if len(trend_df) > 0:
                trend_df["week"] = trend_df["_parsed_time"].dt.to_period("W").apply(lambda r: r.start_time.strftime("%b %d"))
                weekly_missed = trend_df[trend_df["predicted_missed"] == 1].groupby("week").size().reset_index(name="missed")
                weekly_total = trend_df.groupby("week").size().reset_index(name="total")
                weekly = weekly_total.merge(weekly_missed, on="week", how="left").fillna(0)
                weekly["missed"] = weekly["missed"].astype(int)
                weekly = weekly.sort_values("week")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=weekly["week"], y=weekly["total"],
                    mode="lines+markers", name="Total leads",
                    line=dict(color="#8b8f99", width=2), marker=dict(size=8),
                ))
                fig.add_trace(go.Scatter(
                    x=weekly["week"], y=weekly["missed"],
                    mode="lines+markers", name="Missed leads",
                    line=dict(color="#c9503f", width=2), marker=dict(size=8),
                    fill="tozeroy", fillcolor="rgba(201,80,63,0.08)",
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#8b8f99', font_family='IBM Plex Mono, monospace', showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=40, r=40, t=20, b=40),
                    yaxis=dict(showgrid=True, gridcolor='#2c303a', title='Leads'),
                    xaxis=dict(showgrid=False, title='Week'),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data in the past 4 weeks.")
        except Exception:
            st.info("Could not generate trend data.")
    else:
        st.info("Run more scans over time to see trend data.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Leads Saved: Before vs After
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Estimated leads saved</p>", unsafe_allow_html=True)

    saved_col1, saved_col2 = st.columns(2)
    with saved_col1:
        st.markdown(f"""
        <div style="background: var(--flap-rust-dim); border:1px solid rgba(201,80,63,0.2); border-radius:var(--radius-md); padding:1.2rem; text-align:center;">
            <div style="font-size:0.68rem; color:var(--flap-rust); text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:0.5rem; font-family:'IBM Plex Mono',monospace;">Without system</div>
            <div style="font-size:2.3rem; font-weight:700; color:var(--flap-rust); font-family:'IBM Plex Mono',monospace;">{missed_total}</div>
            <div style="font-size:0.8rem; color:var(--ink-dim); margin-top:0.25rem;">leads would have been lost</div>
        </div>
        """, unsafe_allow_html=True)
    with saved_col2:
        st.markdown(f"""
        <div style="background: var(--flap-teal-dim); border:1px solid rgba(79,156,143,0.2); border-radius:var(--radius-md); padding:1.2rem; text-align:center;">
            <div style="font-size:0.68rem; color:var(--flap-teal); text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:0.5rem; font-family:'IBM Plex Mono',monospace;">With system</div>
            <div style="font-size:2.3rem; font-weight:700; color:var(--flap-teal); font-family:'IBM Plex Mono',monospace;">{leads_saved}</div>
            <div style="font-size:0.8rem; color:var(--ink-dim); margin-top:0.25rem;">leads recovered via auto-reply</div>
        </div>
        """, unsafe_allow_html=True)

    if HAS_PLOTLY and missed_total > 0:
        fig = go.Figure(data=[
            go.Bar(name='Without system', x=['Leads'], y=[missed_total], marker_color='#c9503f'),
            go.Bar(name='Recovered', x=['Leads'], y=[leads_saved], marker_color='#4f9c8f'),
        ])
        fig.update_layout(
            barmode='group', title='Leads lost vs leads recovered',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#8b8f99', font_family='IBM Plex Mono, monospace', showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(showgrid=True, gridcolor='#2c303a'),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif missed_total == 0:
        st.info("No missed leads yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Recovery Details
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Recovery breakdown</p>", unsafe_allow_html=True)

    if missed_total > 0:
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        with detail_col1:
            auto_pct = (auto_reply_count / missed_total * 100) if missed_total > 0 else 0
            st.metric(label="Auto-replied", value=auto_reply_count, delta=f"{auto_pct:.0f}% of missed")
        with detail_col2:
            human_pct = (recovered / missed_total * 100) if missed_total > 0 else 0
            st.metric(label="Human follow-up", value=recovered, delta=f"{human_pct:.0f}% of missed")
        with detail_col3:
            pending = max(0, missed_total - auto_reply_count - recovered)
            st.metric(label="Still pending", value=pending,
                      delta="Needs attention" if pending > 0 else "All handled",
                      delta_color="inverse" if pending > 0 else "normal")

        if HAS_PLOTLY:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=recovery_rate,
                title={"text": "Recovery rate (%)", "font": {"color": "#eae7dd", "family": "IBM Plex Mono, monospace"}},
                number={"suffix": "%", "font": {"color": "#eae7dd", "family": "IBM Plex Mono, monospace"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b8f99"},
                    "bar": {"color": "#4f9c8f"},
                    "bgcolor": "#1c1f26",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(201,80,63,0.12)"},
                        {"range": [50, 80], "color": "rgba(232,163,61,0.12)"},
                        {"range": [80, 100], "color": "rgba(79,156,143,0.12)"},
                    ],
                    "threshold": {"line": {"color": "#e8a33d", "width": 3}, "thickness": 0.8, "value": recovery_rate},
                },
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#8b8f99', height=300,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missed leads to analyze yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: Workflow Settings
# ══════════════════════════════════════════════════════════
elif page == "Workflow Settings":
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Workflow configuration &amp; business settings</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--ink-dim); margin-top:-0.5rem;'>Update templates, courses, placement rates, and notification boundaries. Changes save immediately.</p>", unsafe_allow_html=True)

    save_status = st.empty()

    with st.form("settings_form"):
        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("##### Business identity")
            company_name = st.text_input("Company name", overrides.get("COMPANY_NAME", config.COMPANY_NAME))
            sender_name = st.text_input("Sender name", overrides.get("SENDER_NAME", config.SENDER_NAME))
            team_phone = st.text_input("Contact phone", overrides.get("TEAM_PHONE", config.TEAM_PHONE))
            team_email = st.text_input("Contact email", overrides.get("TEAM_EMAIL", config.TEAM_EMAIL))
            website_url = st.text_input("Website URL", overrides.get("WEBSITE_URL", config.WEBSITE_URL))

            st.markdown("##### Escalation rules")
            overdue_hrs = st.number_input("Overdue alert (hours)", value=int(overrides.get("HOURS_BEFORE_OVERDUE", config.HOURS_BEFORE_OVERDUE)), min_value=1)
            esc_hrs = st.number_input("Escalation (hours)", value=int(overrides.get("HOURS_BEFORE_ESCALATION", config.HOURS_BEFORE_ESCALATION)), min_value=1)

            st.markdown("##### Recovery rate alert")
            recovery_threshold = st.slider(
                "Alert when recovery rate drops below (%)", min_value=10, max_value=100,
                value=int(overrides.get("RECOVERY_RATE_THRESHOLD", config.RECOVERY_RATE_THRESHOLD)),
                help="Email alert is sent when fewer than this % of missed leads are handled.",
            )

        with sc2:
            st.markdown("##### Course offerings")
            placement_rate = st.text_input("Placement rate", overrides.get("PLACEMENT_RATE", config.PLACEMENT_RATE))
            partners = st.text_input("Hiring partners", overrides.get("COMPANY_PARTNERS", config.COMPANY_PARTNERS))
            discount = st.text_area("Discount text", overrides.get("DISCOUNT_INFO", config.DISCOUNT_INFO), height=80)
            scholarship = st.text_area("Scholarship info", overrides.get("SCHOLARSHIP_INFO", config.SCHOLARSHIP_INFO), height=80)
            emi_text = st.text_area("EMI details", overrides.get("EMI_INFO", config.EMI_INFO), height=80)

            st.markdown("##### Reply delay (seconds)")
            td1, td2 = st.columns(2)
            with td1:
                min_delay = st.number_input("Min delay", value=int(overrides.get("MIN_REPLY_DELAY", config.MIN_REPLY_DELAY)), min_value=1)
            with td2:
                max_delay = st.number_input("Max delay", value=int(overrides.get("MAX_REPLY_DELAY", config.MAX_REPLY_DELAY)), min_value=1)

        st.markdown("##### Email signature")
        sig_val = overrides.get("EMAIL_SIGNATURE", f"Best regards,\n{sender_name}\n{company_name}\nPhone: {team_phone}\nEmail: {team_email}\nWeb: {website_url}")
        email_signature = st.text_area("Email signature block", sig_val, height=120)

        submitted = st.form_submit_button("Save settings", type="primary")

        if submitted:
            new_overrides = overrides.copy()
            new_overrides.update({
                "COMPANY_NAME": company_name, "SENDER_NAME": sender_name,
                "TEAM_PHONE": team_phone, "TEAM_EMAIL": team_email,
                "WEBSITE_URL": website_url, "HOURS_BEFORE_OVERDUE": overdue_hrs,
                "HOURS_BEFORE_ESCALATION": esc_hrs, "RECOVERY_RATE_THRESHOLD": recovery_threshold,
                "PLACEMENT_RATE": placement_rate, "COMPANY_PARTNERS": partners,
                "DISCOUNT_INFO": discount, "SCHOLARSHIP_INFO": scholarship,
                "EMI_INFO": emi_text, "MIN_REPLY_DELAY": min_delay,
                "MAX_REPLY_DELAY": max_delay, "EMAIL_SIGNATURE": email_signature,
            })
            save_overrides(new_overrides)
            save_status.success("Settings saved.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Environment Variables Panel
    st.markdown("<div class='bd-card'>", unsafe_allow_html=True)
    st.markdown("<p class='bd-card-title'>Environment variables</p>", unsafe_allow_html=True)

    env_list = {
        "SMTP_USER": _get_secret("SMTP_USER"),
        "SMTP_HOST": "smtp.gmail.com",
        "IMAP_USER": _get_secret("IMAP_USER"),
        "NOTIFY_EMAIL": _get_secret("NOTIFY_EMAIL"),
        "SENDER_NAME": _get_secret("SENDER_NAME"),
    }

    for key, val in env_list.items():
        val_str = "Not set" if not val else val
        st.markdown(f"**{key}:** `{val_str}`")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="bd-footer">
    MISSED-LEAD OPS BOARD · AI-POWERED SALES DISPATCH · CIT CHENNAI
</div>
""", unsafe_allow_html=True)