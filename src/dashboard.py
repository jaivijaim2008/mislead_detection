"""
dashboard.py — Missed-Lead Detector
Solar Flare Command Center — Unique warm-ember analytics interface.

Complete redesign: solar-warm palette, ember/sun/gold gradients,
holographic depth, animated heat shimmer, and smooth micro-interactions.

Design: Deep space base with warm solar flare accents, amber glow effects,
animated gradient borders, and faceted glass elements.
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
    page_title="Missed-Lead Command Center",
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
    /* Hide sidebar and header during login */
    [data-testid="stSidebar"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .stApp {
        background: linear-gradient(135deg, #0b0710 0%, #110d1a 50%, #0d0916 100%) !important;
        display: flex; align-items: center; justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered login card
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style="
            background: rgba(17,13,26,0.85);
            border: 1px solid rgba(255,140,50,0.15);
            border-radius: 20px;
            padding: 2.8rem 2.5rem 2.2rem;
            margin-top: 8vh;
            box-shadow: 0 0 60px rgba(255,140,50,0.06);
            font-family: 'Outfit', sans-serif;
        ">
            <div style="text-align:center; margin-bottom: 2rem;">
                <div style="font-size:2.8rem; margin-bottom:0.5rem;">🔐</div>
                <h2 style="color:#e8ecf1; margin:0; font-weight:700; font-size:1.6rem;">Missed-Lead Detector</h2>
                <p style="color:#6b7a90; margin:0.4rem 0 0; font-size:0.9rem;">Sign in to access the Command Center</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                if username.strip() == _AUTH_USER and password == _AUTH_PASS:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")
    st.stop()  # Block everything below until authenticated

# ══════════════════════════════════════════════════════════
#  SOLAR FLARE DESIGN SYSTEM — CSS
# ══════════════════════════════════════════════════════════
DESIGN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --flare-bg: #0b0710;
        --flare-surface: #110d1a;
        --flare-card: rgba(17, 13, 26, 0.75);
        --flare-border: rgba(255, 140, 50, 0.08);
        --flare-border-hover: rgba(255, 140, 50, 0.25);
        --sun-glow: #ff8c32;
        --sun-glow-dim: rgba(255, 140, 50, 0.12);
        --ember: #ff5e3a;
        --ember-dim: rgba(255, 94, 58, 0.12);
        --gold: #ffc857;
        --gold-dim: rgba(255, 200, 87, 0.12);
        --rose: #ff6b8a;
        --rose-dim: rgba(255, 107, 138, 0.12);
        --teal: #2dd4bf;
        --teal-dim: rgba(45, 212, 191, 0.12);
        --text-primary: #f0ece4;
        --text-secondary: #8a7f94;
        --text-muted: #4a3f54;
        --radius-sm: 6px;
        --radius-md: 14px;
        --radius-lg: 22px;
        --radius-pill: 999px;
    }

    html, body, [class*="st-"] {
        font-family: 'Outfit', -apple-system, sans-serif !important;
    }

    .stApp {
        background: var(--flare-bg) !important;
        background-image:
            radial-gradient(ellipse at 80% 20%, rgba(255, 140, 50, 0.04) 0%, transparent 60%),
            radial-gradient(ellipse at 20% 80%, rgba(255, 94, 58, 0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(45, 212, 191, 0.015) 0%, transparent 40%) !important;
        color: var(--text-primary) !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 140, 50, 0.15); border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 140, 50, 0.3); }

    @keyframes flare-floatUp {
        from { opacity: 0; transform: translateY(25px) scale(0.96); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes flare-slideRight {
        from { opacity: 0; transform: translateX(-25px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes flare-scaleBounce {
        from { opacity: 0; transform: scale(0.85) rotate(-2deg); }
        60%  { transform: scale(1.03) rotate(0.5deg); }
        to   { opacity: 1; transform: scale(1) rotate(0); }
    }
    @keyframes flare-pulseGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(255, 140, 50, 0.03); }
        50%      { box-shadow: 0 0 35px rgba(255, 140, 50, 0.08); }
    }
    @keyframes flare-sunRise {
        0%   { background-position: 0% 0%; opacity: 0.3; }
        50%  { background-position: 100% 100%; opacity: 0.7; }
        100% { background-position: 0% 0%; opacity: 0.3; }
    }
    @keyframes flare-heatShimmer {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes flare-warmPulse {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50%      { transform: scale(1.05); opacity: 1; }
    }
    @keyframes flare-pulseDot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.3; transform: scale(0.6); }
    }

    .flare-animate-in {
        animation: flare-floatUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .flare-animate-in-delay-1 { animation-delay: 0.08s; }
    .flare-animate-in-delay-2 { animation-delay: 0.16s; }
    .flare-animate-in-delay-3 { animation-delay: 0.24s; }
    .flare-animate-in-delay-4 { animation-delay: 0.32s; }
    .flare-animate-in-delay-5 { animation-delay: 0.4s; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0916 0%, #110d1e 50%, #0d0916 100%) !important;
        border-right: 1px solid var(--flare-border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--flare-border) !important;
    }

    .neb-header {
        position: relative;
        background: linear-gradient(135deg, rgba(17, 13, 26, 0.8), rgba(25, 15, 25, 0.8));
        backdrop-filter: blur(16px);
        border: 1px solid var(--flare-border);
        border-radius: var(--radius-lg);
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 2rem;
        overflow: hidden;
        animation: flare-pulseGlow 5s ease-in-out infinite;
    }
    .neb-header::before {
        content: '';
        position: absolute;
        top: -150px; right: -100px;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(255, 140, 50, 0.06) 0%, rgba(255, 94, 58, 0.03) 30%, transparent 60%);
        animation: flare-sunRise 15s ease-in-out infinite;
        background-size: 200% 200%;
        pointer-events: none;
    }
    .neb-header::after {
        content: '';
        position: absolute;
        bottom: -80px; left: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(45, 212, 191, 0.03) 0%, transparent 70%);
        pointer-events: none;
    }
    .neb-header h1 {
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--sun-glow) 40%, var(--ember) 70%, var(--gold) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .neb-header p {
        color: var(--text-secondary);
        margin-top: 0.4rem;
        font-size: 0.95rem;
        font-weight: 300;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }

    .neb-card {
        position: relative;
        background: var(--flare-card);
        backdrop-filter: blur(14px);
        border: 1px solid var(--flare-border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        animation: flare-floatUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .neb-card:hover {
        border-color: var(--flare-border-hover);
        transform: translateY(-3px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(255, 140, 50, 0.04);
    }

    .neb-card-glow {
        position: relative;
        background: var(--flare-card);
        backdrop-filter: blur(14px);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        animation: flare-floatUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .neb-card-glow::before {
        content: '';
        position: absolute;
        inset: -1px;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, var(--sun-glow), var(--rose), var(--ember), var(--gold), var(--sun-glow));
        background-size: 400% 400%;
        animation: flare-heatShimmer 8s ease infinite;
        z-index: -1;
        opacity: 0.5;
    }
    .neb-card-glow::after {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: calc(var(--radius-md) - 1px);
        background: var(--flare-card);
        z-index: -1;
    }

    .neb-kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .neb-kpi {
        flex: 1;
        background: var(--flare-card);
        backdrop-filter: blur(14px);
        border: 1px solid var(--flare-border);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
        transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .neb-kpi:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }
    .neb-kpi::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: var(--radius-md) var(--radius-md) 0 0;
    }
    .neb-kpi::after {
        content: '';
        position: absolute;
        top: -30px; right: -30px;
        width: 100px; height: 100px;
        border-radius: 50%;
        opacity: 0.05;
        transition: all 0.6s ease;
    }
    .neb-kpi:hover::after {
        transform: scale(2);
        opacity: 0.12;
    }

    .neb-kpi.sun::before    { background: linear-gradient(90deg, var(--sun-glow), #e07c30); }
    .neb-kpi.sun::after     { background: var(--sun-glow); }
    .neb-kpi.ember::before  { background: linear-gradient(90deg, var(--ember), #d6442a); }
    .neb-kpi.ember::after   { background: var(--ember); }
    .neb-kpi.teal::before   { background: linear-gradient(90deg, var(--teal), #1fa893); }
    .neb-kpi.teal::after    { background: var(--teal); }
    .neb-kpi.gold::before   { background: linear-gradient(90deg, var(--gold), #d4a032); }
    .neb-kpi.gold::after    { background: var(--gold); }
    .neb-kpi.rose::before   { background: linear-gradient(90deg, var(--rose), #d94a6a); }
    .neb-kpi.rose::after    { background: var(--rose); }

    .neb-kpi-icon {
        font-size: 1.4rem;
        margin-bottom: 0.4rem;
        display: inline-block;
        animation: flare-warmPulse 4s ease-in-out infinite;
    }
    .neb-kpi-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.03em;
        font-family: 'JetBrains Mono', monospace;
    }
    .neb-kpi-label {
        font-size: 0.7rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .neb-kpi.sun .neb-kpi-value   { color: var(--sun-glow); }
    .neb-kpi.ember .neb-kpi-value { color: var(--ember); }
    .neb-kpi.teal .neb-kpi-value  { color: var(--teal); }
    .neb-kpi.gold .neb-kpi-value  { color: var(--gold); }
    .neb-kpi.rose .neb-kpi-value  { color: var(--rose); }

    .neb-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.85rem;
        border-radius: var(--radius-pill);
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .neb-badge-live {
        background: var(--teal-dim);
        color: var(--teal);
        border: 1px solid rgba(45, 212, 191, 0.2);
    }
    .neb-badge-live::before {
        content: '';
        width: 6px; height: 6px;
        background: var(--teal);
        border-radius: 50%;
        animation: flare-pulseDot 1.8s infinite;
    }
    .neb-badge-demo {
        background: var(--gold-dim);
        color: var(--gold);
        border: 1px solid rgba(255, 200, 87, 0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, rgba(17, 13, 26, 0.9), rgba(25, 18, 30, 0.9)) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--flare-border) !important;
        border-radius: var(--radius-pill) !important;
        padding: 0.5rem 1.6rem !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 0.02em !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, var(--sun-glow), var(--ember));
        opacity: 0;
        transition: opacity 0.4s ease;
        border-radius: inherit;
    }
    .stButton > button:hover {
        border-color: var(--sun-glow) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 25px rgba(255, 140, 50, 0.15), 0 0 50px rgba(255, 140, 50, 0.04) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:hover::before {
        opacity: 0.12;
    }
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--sun-glow), var(--ember)) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 25px rgba(255, 140, 50, 0.25) !important;
    }
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #ffa050, var(--sun-glow)) !important;
        box-shadow: 0 6px 35px rgba(255, 140, 50, 0.35) !important;
    }

    .neb-btn-danger > button {
        background: linear-gradient(135deg, rgba(255, 94, 58, 0.1), rgba(214, 68, 42, 0.1)) !important;
        border-color: rgba(255, 94, 58, 0.25) !important;
        color: var(--ember) !important;
    }
    .neb-btn-danger > button:hover {
        border-color: var(--ember) !important;
        box-shadow: 0 4px 20px rgba(255, 94, 58, 0.15) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    textarea {
        background: rgba(17, 13, 26, 0.6) !important;
        border: 1px solid var(--flare-border) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'Outfit', sans-serif !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] input:hover,
    textarea:hover {
        border-color: var(--flare-border-hover) !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] input:focus,
    textarea:focus {
        border-color: var(--sun-glow) !important;
        box-shadow: 0 0 0 3px rgba(255, 140, 50, 0.06) !important;
    }

    .neb-notif {
        background: rgba(17, 13, 26, 0.5);
        border: 1px solid var(--flare-border);
        border-left: 3px solid var(--sun-glow);
        border-radius: var(--radius-sm);
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        transition: all 0.3s ease;
        animation: flare-slideRight 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .neb-notif:hover {
        background: rgba(25, 18, 35, 0.5);
        border-left-color: var(--sun-glow);
        transform: translateX(5px);
    }
    .neb-notif.new_lead   { border-left-color: var(--sun-glow); }
    .neb-notif.auto_reply { border-left-color: var(--teal); }
    .neb-notif.overdue    { border-left-color: var(--ember); }
    .neb-notif.info       { border-left-color: var(--rose); }

    .neb-terminal {
        background: #07040d !important;
        border: 1px solid var(--flare-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 1.2rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        color: var(--gold) !important;
        box-shadow: inset 0 2px 15px rgba(0, 0, 0, 0.5) !important;
        line-height: 1.6 !important;
        max-height: 400px;
        overflow-y: auto;
    }

    .neb-email {
        background: var(--flare-card);
        border: 1px solid var(--flare-border);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        margin-top: 1rem;
        animation: flare-scaleBounce 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .neb-email-header {
        border-bottom: 1px solid var(--flare-border);
        padding-bottom: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    .neb-email-header span { color: var(--text-primary); font-weight: 500; }
    .neb-email-body {
        font-size: 0.88rem;
        color: var(--text-primary);
        line-height: 1.6;
        white-space: pre-wrap;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: var(--flare-card);
        border-radius: var(--radius-sm);
        padding: 0.3rem;
        border: 1px solid var(--flare-border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary) !important;
        font-weight: 500;
        font-family: 'Outfit', sans-serif;
        padding: 0.35rem 1.1rem;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--sun-glow-dim) !important;
        color: var(--sun-glow) !important;
        border: 1px solid rgba(255, 140, 50, 0.15) !important;
    }

    [data-testid="stDialog"] {
        background: var(--flare-card) !important;
        border: 1px solid var(--flare-border) !important;
        border-radius: var(--radius-lg) !important;
        backdrop-filter: blur(16px) !important;
    }

    .neb-sidebar-box {
        background: rgba(17, 13, 26, 0.5);
        border: 1px solid var(--flare-border);
        border-radius: var(--radius-sm);
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        line-height: 1.8;
    }

    #MainMenu, footer { visibility: hidden; }

    .neb-footer {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        color: var(--text-muted);
        font-size: 0.72rem;
        border-top: 1px solid var(--flare-border);
        margin-top: 4rem;
        letter-spacing: 0.06em;
    }

    [data-testid="stMetric"] {
        background: var(--flare-card);
        border: 1px solid var(--flare-border);
        border-radius: var(--radius-sm);
        padding: 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--flare-border-hover);
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }

    @media (max-width: 768px) {
        .neb-header {
            padding: 1.5rem 1.25rem 1.25rem;
            border-radius: var(--radius-md);
        }
        .neb-header h1 { font-size: 1.5rem; }
        .neb-header p { font-size: 0.85rem; }
        .neb-kpi-row { flex-wrap: wrap; }
        .neb-kpi {
            flex: 1 1 calc(50% - 0.5rem);
            min-width: calc(50% - 0.5rem);
            padding: 1rem;
        }
        .neb-kpi-value { font-size: 1.4rem; }
        .neb-card { padding: 1.25rem; border-radius: var(--radius-sm); }
        .neb-footer { padding: 1.5rem 1rem 1rem; }
    }
    @media (max-width: 480px) {
        .neb-kpi { flex: 1 1 100%; min-width: 100%; }
        .neb-header h1 { font-size: 1.3rem; }
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
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin:0; color: var(--text-primary); font-weight: 700;">
            {'🔴' if lead_data.get('predicted_missed') == 1 else '🟢'} Lead {lead_data.get('lead_id', 'N/A')}
        </h2>
        <p style="color: var(--text-secondary); margin-top: 0.25rem; font-size: 0.9rem;">
            Full lead inspection and smart reply preview
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 📋 Lead Information")
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
            ("Response Gap", f"**{lead_data.get('response_gap_hrs', 0):.1f}** hours"),
            ("Risk Score", f"**{lead_data.get('missed_probability', 0):.1%}**"),
            ("Status", f"{'🔴 MISSED LEAD' if lead_data.get('predicted_missed') == 1 else '🟢 Responded'}"),
        ])
        for label, value in info_items:
            st.markdown(f"**{label}:** {value}")

        st.markdown("---")
        st.markdown("#### 💬 Original Message")
        st.info(lead_data.get('message_text', 'No message content available.'))

    with c2:
        st.markdown("#### 🤖 Smart Auto-Reply Preview")
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
            <span class="neb-badge" style="background: var(--sun-glow-dim); color: var(--sun-glow); border: 1px solid rgba(255,140,50,0.2); margin-bottom: 1rem;">
                Intent: {reply_payload['detected_intent'].upper()}
            </span>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="neb-email">
                <div class="neb-email-header">
                    <div>From: <span>Sales Team &lt;noreply@yourcompany.com&gt;</span></div>
                    <div>To: <span>{lead_data.get('_customer_name', 'Valued Customer')} &lt;{lead_data.get('_customer_email', 'customer@example.com')}&gt;</span></div>
                    <div>Subject: <span>{reply_payload["reply_subject"]}</span></div>
                </div>
                <div class="neb-email-body">{reply_payload["reply_body"]}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not load smart reply templates: {e}")


# ── Header Section ─────────────────────────────────────────
live_connected = is_live_mode()
status_label = "Gmail Connected — Live" if live_connected else "Gmail Not Connected"
status_class = "neb-badge-live" if live_connected else "neb-badge-demo"

st.markdown(f"""
<div class="neb-header flare-animate-in">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <h1>Missed-Lead Command Center</h1>
        <span class="neb-badge {status_class}">{status_label}</span>
    </div>
    <p>AI-Powered Email Monitoring &bull; Smart Auto-Replies &bull; Sales Pipeline Retention</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Configurations & Navigation ───────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Go to:",
        ["Command Center", "Lead Explorer", "Auto-Replies Tracker",
         "Interactive Pipeline Graph", "Performance Overview", "Workflow Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### System Health")

    leads_df = load_scored_leads()
    notifs = load_json_log(NOTIF_LOG)
    unread_notifs = [n for n in notifs if not n.get("read", False)] if isinstance(notifs, list) else []

    inbox_icon = "🟢" if live_connected else "🔴"
    st.markdown(f"""
    <div class="neb-sidebar-box">
        <div><b>Gmail</b>: {inbox_icon} {'Active' if live_connected else 'Off'}</div>
        <div><b>Scored</b>: {len(leads_df)} leads</div>
        <div><b>Alerts</b>: <span style="color: {'var(--ember)' if len(unread_notifs) > 0 else 'var(--text-secondary)'}; font-weight: bold;">{len(unread_notifs)}</span></div>
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
        <div class="neb-card-glow" style="text-align:center; padding: 3rem 2rem;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem; animation: flare-warmPulse 3s ease-in-out infinite;">📬</div>
            <h2 style="color: var(--text-primary); margin-bottom: 0.5rem; font-weight: 700;">No Inbox Data Yet</h2>
            <p style="color: var(--text-secondary); font-size: 1rem; max-width: 520px; margin: 0 auto;">
                Connect your Gmail account to start scanning real customer emails.
            </p>
            <div style="margin-top: 1.5rem; padding: 1.25rem; background: rgba(17,13,26,0.5); border-radius: var(--radius-md); border: 1px solid var(--flare-border); max-width: 550px; margin-left: auto; margin-right: auto; text-align: left;">
                <p style="color: var(--gold); font-weight: 600; margin-bottom: 0.75rem;">⚡ Setup Steps:</p>
                <p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 2;">
                    1. Enable 2FA on your Gmail and generate an App Password<br>
                    2. Add these to Streamlit Cloud secrets:<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--sun-glow);">IMAP_USER</code> = your Gmail<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--sun-glow);">IMAP_PASS</code> = App Password<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--sun-glow);">SMTP_USER</code> = your Gmail<br>
                    &nbsp;&nbsp;&nbsp;• <code style="color:var(--sun-glow);">SMTP_PASS</code> = App Password<br>
                    3. Click "Trigger Scan Now" to fetch emails
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KPI Row ──────────────────────────────────────────
    st.markdown(f"""
    <div class="neb-kpi-row">
        <div class="neb-kpi sun flare-animate-in flare-animate-in-delay-1">
            <div class="neb-kpi-icon">📊</div>
            <div class="neb-kpi-value">{total_leads:,}</div>
            <div class="neb-kpi-label">Scanned Leads</div>
        </div>
        <div class="neb-kpi ember flare-animate-in flare-animate-in-delay-2">
            <div class="neb-kpi-icon">🚨</div>
            <div class="neb-kpi-value">{missed_leads}</div>
            <div class="neb-kpi-label">Missed Leads</div>
        </div>
        <div class="neb-kpi teal flare-animate-in flare-animate-in-delay-3">
            <div class="neb-kpi-icon">🤖</div>
            <div class="neb-kpi-value">{replied_count}</div>
            <div class="neb-kpi-label">Auto-Replied</div>
        </div>
        <div class="neb-kpi gold flare-animate-in flare-animate-in-delay-4">
            <div class="neb-kpi-icon">⏳</div>
            <div class="neb-kpi-value">{overdue_count}</div>
            <div class="neb-kpi-label">Awaiting Action</div>
        </div>
        <div class="neb-kpi rose flare-animate-in flare-animate-in-delay-5">
            <div class="neb-kpi-icon">🔥</div>
            <div class="neb-kpi-value">{high_intent}</div>
            <div class="neb-kpi-label">High Intent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Layout ──────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Pipeline Controls
        st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
        st.markdown("#### 📧 Inbox Monitoring Pipeline")

        c1, c2 = st.columns(2)
        with c1:
            scan_btn = st.button("⚡ Trigger Scan Now", type="primary", use_container_width=True)
        with c2:
            dry_btn = st.button("🧪 Simulate Dry-Run", use_container_width=True)

        if scan_btn or dry_btn:
            is_dry = bool(dry_btn)
            st.markdown("**Pipeline Output:**")
            with st.spinner("Processing mailbox — scanning Gmail..."):
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
                    st.markdown(f"<pre class='neb-terminal'>{stdout}</pre>", unsafe_allow_html=True)
                    if res.stderr:
                        with st.expander("⚠️ Warnings / stderr"):
                            st.code(res.stderr)
                    st.cache_data.clear()
                    st.rerun()
                except subprocess.TimeoutExpired:
                    st.error("⏱️ Scan timed out after 120 seconds. Gmail may be slow — try again.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")
        elif not live_connected:
            st.warning("⚠️ Gmail not connected. Set IMAP_USER and IMAP_PASS in Streamlit secrets.")
        else:
            st.markdown("<p style='color: var(--text-secondary); font-style: italic;'>Trigger a pipeline scan to read Gmail inboxes, run ML predictions, and execute automatic replies.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Distribution Chart
        st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Lead Risk Distribution")

        if total_leads > 0 and "missed_probability" in scored.columns:
            if HAS_PLOTLY:
                fig = px.histogram(
                    scored, x="missed_probability", nbins=20,
                    title="Lead Risk Distribution (Threshold = 0.50)",
                    labels={"missed_probability": "Predicted Missed Probability", "count": "Lead Count"},
                    color_discrete_sequence=["#ff6b6b"],
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#6b7a90',
                    title_font_color='#e8ecf1',
                    showlegend=False,
                    xaxis=dict(showgrid=False, linecolor='rgba(0,212,255,0.06)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,212,255,0.06)', linecolor='rgba(0,212,255,0.06)'),
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="#fbbf24",
                              annotation_text="Threshold", annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)
            else:
                if os.path.exists(CM_IMG) and HAS_PIL:
                    st.image(Image.open(CM_IMG), use_container_width=True)
        else:
            st.info("No lead metrics scored. Trigger a scan above to ingest data.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='neb-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("#### 🔔 Notification Feed")

        if isinstance(notifs, list) and notifs:
            if len(unread_notifs) > 0:
                if st.button(f"✓ Mark all read ({len(unread_notifs)})", use_container_width=True):
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
                    <div class="neb-notif {ntype}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
                            <span style="font-weight: 600; font-size: 0.88rem; color: var(--text-primary);">{icon} {n.get('title', 'Notification')}</span>
                            <span style="font-size: 0.7rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">{n.get('timestamp', '')[11:16]}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4; margin-top: 0.2rem;">{n.get('message', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    unread_count += 1
            if unread_count == 0:
                st.markdown("<p style='text-align:center; color:var(--text-muted); padding: 2rem 0;'>All caught up! ✨</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center; color:var(--text-muted); padding: 2rem 0;'>No notifications yet.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: Lead Explorer
# ══════════════════════════════════════════════════════════
elif page == "Lead Explorer":
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Interactive Lead Table & Inspector")

    scored = load_scored_leads()

    if scored.empty:
        st.warning("No leads recorded. Please trigger an inbox scan to fetch records.")
    else:
        has_gmail_headers = "_customer_name" in scored.columns

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            q_search = st.text_input("🔍 Search Customer / Email", "")
        with c2:
            q_status = st.selectbox("Pipeline Status", ["All", "Missed Leads", "Responded (Low risk)"])
        with c3:
            q_intent = st.selectbox("High Intent", ["All", "Yes", "No"])
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
            val = 1 if q_status == "Missed Leads" else 0
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
                "channel": "Source", "response_gap_hrs": "Gap (Hrs)",
                "high_intent_flag": "High Intent", "missed_probability": "Risk Score",
                "predicted_missed": "Missed",
            }
        else:
            show_cols = ["lead_id", "channel", "message_text", "response_gap_hrs",
                         "high_intent_flag", "missed_probability", "predicted_missed"]
            show_cols = [c for c in show_cols if c in filtered.columns]
            rename_map = {
                "lead_id": "Lead ID", "channel": "Source",
                "message_text": "Inquiry", "response_gap_hrs": "Gap (Hrs)",
                "high_intent_flag": "High Intent", "missed_probability": "Risk Score",
                "predicted_missed": "Missed",
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
        st.markdown("#### 🔍 Lead Inspector")

        selected_id = st.selectbox("Select a Lead ID:", ["-- None --"] + list(filtered["lead_id"].unique()))

        if selected_id != "-- None --":
            lead_row = filtered[filtered["lead_id"] == selected_id].iloc[0].to_dict()

            d_col1, d_col2 = st.columns([1, 1])

            with d_col1:
                st.markdown("<div style='background:rgba(17,13,26,0.4); padding:1.25rem; border-radius:var(--radius-sm); border:1px solid var(--flare-border);'>", unsafe_allow_html=True)
                st.markdown(f"**Lead ID:** `{lead_row.get('lead_id', 'N/A')}`")

                if has_gmail_headers:
                    st.markdown(f"**Customer:** {lead_row.get('_customer_name', 'N/A')}")
                    st.markdown(f"**Email:** `{lead_row.get('_customer_email', 'N/A')}`")
                    st.markdown(f"**Subject:** {lead_row.get('_subject', 'N/A')}")
                    st.markdown(f"**Received:** {lead_row.get('_received_time', 'Unknown')}")
                else:
                    st.markdown(f"**Source:** {lead_row.get('channel', 'N/A')}")

                st.markdown(f"**Response Gap:** {lead_row.get('response_gap_hrs', 0):.1f} hours")
                st.markdown(f"**Risk Score:** `{lead_row.get('missed_probability', 0):.2f}`")
                st.markdown(f"**Status:** {'🔴 MISSED LEAD' if lead_row.get('predicted_missed') == 1 else '🟢 Responded'}")

                st.markdown("**Original Message:**")
                st.info(lead_row.get("message_text", "No message"))
                st.markdown("</div>", unsafe_allow_html=True)

            with d_col2:
                st.markdown("**Smart Auto-Reply Preview:**")
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
                    <span class="neb-badge" style="background: var(--sun-glow-dim); color: var(--sun-glow); border: 1px solid rgba(255,140,50,0.2); margin-bottom: 1rem;">
                        Intent: {reply_payload['detected_intent'].upper()}
                    </span>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="neb-email">
                        <div class="neb-email-header">
                            <div>From: <span>Sales Team &lt;noreply@yourcompany.com&gt;</span></div>
                            <div>To: <span>{lead_row.get('_customer_name', 'Valued Customer')} &lt;{lead_row.get('_customer_email', 'customer@example.com')}&gt;</span></div>
                            <div>Subject: <span>{reply_payload["reply_subject"]}</span></div>
                        </div>
                        <div class="neb-email-body">{reply_payload["reply_body"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not load smart reply templates: {e}")

            # Open in Modal button
            if st.button("🔎 Open Full Detail Modal", use_container_width=True):
                show_lead_dialog(lead_row, has_gmail_headers)
        else:
            st.info("Pick a lead ID above to drill down into details.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: Auto-Replies Tracker
# ══════════════════════════════════════════════════════════
elif page == "Auto-Replies Tracker":
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 📬 Automated Auto-Replies History")

    reply_log = load_json_log(REPLY_LOG)
    followup_status = load_json_log(FOLLOWUP_LOG)

    if not reply_log:
        st.info("No auto-replies logged yet. They are generated when the inbox monitoring detects new missed leads.")
    else:
        reply_df = pd.DataFrame(reply_log)
        st.markdown("---")
        st.markdown("#### Sent Reply Logs")
        show_cols = [c for c in ["lead_id", "customer_name", "customer_email", "reply_subject", "detected_intent", "replied_at"] if c in reply_df.columns]
        st.dataframe(reply_df[show_cols], use_container_width=True)

        st.markdown("---")
        st.markdown("#### ⌛ Overdue Human Follow-Up")

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
                        "alert_escalated": "🔥 Overdue" if info.get("overdue_notified") else "—",
                    })

            fu_df = pd.DataFrame(fu_items)
            pending_leads = [item["lead_id"] for item in fu_items if "Pending" in item["human_follow_up"]]

            if pending_leads:
                col_sel, col_act = st.columns([2, 1])
                with col_sel:
                    action_id = st.selectbox("Mark Lead as Resolved", ["-- Select --"] + pending_leads)
                with col_act:
                    st.markdown("<div style='margin-top:1.75rem;'></div>", unsafe_allow_html=True)
                    if st.button("✅ Complete Human Action", use_container_width=True) and action_id != "-- Select --":
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
#  PAGE: Interactive Pipeline Graph
# ══════════════════════════════════════════════════════════
elif page == "Interactive Pipeline Graph":
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 🌊 Lead Flow Pipeline")

    scored = load_scored_leads()
    reply_log = load_json_log(REPLY_LOG)

    if len(scored) == 0:
        st.info("No leads available to visualize.")
    else:
        st.markdown("<p style='color: var(--text-secondary);'>Interactive visualization of how leads flow through your sales pipeline.</p>", unsafe_allow_html=True)
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
                    line=dict(color="rgba(0,212,255,0.08)", width=0.5),
                    label=["Total Leads", "Responded (Safe)", "Missed Leads",
                           "High Intent (Missed)", "Low Intent (Missed)",
                           "Auto-Replied", "Awaiting Human"],
                    color=["#00d4ff", "#10b981", "#ff6b6b", "#fbbf24", "#3d4a5c", "#10b981", "#ff6b6b"],
                ),
                link=dict(
                    source=[0, 0, 2, 2, 3, 3],
                    target=[1, 2, 3, 4, 5, 6],
                    value=[max(responded, 1), max(missed, 1), max(high_intent_missed, 1),
                           max(low_intent_missed, 1), max(auto_replied, 1), max(awaiting, 1)],
                    color=["rgba(16,185,129,0.2)", "rgba(255,107,107,0.2)", "rgba(251,191,36,0.2)",
                           "rgba(61,74,92,0.2)", "rgba(16,185,129,0.2)", "rgba(255,107,107,0.2)"],
                ),
            )])
            fig.update_layout(
                title_text="Customer Journey & Sales Bottlenecks",
                font_size=14, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', font_color='#e8ecf1', height=600,
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

    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 📈 Performance Overview")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-0.5rem;'>Your sales lead recovery at a glance.</p>", unsafe_allow_html=True)

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
    <div class="neb-kpi-row">
        <div class="neb-kpi ember flare-animate-in flare-animate-in-delay-1">
            <div class="neb-kpi-icon">🚨</div>
            <div class="neb-kpi-value">{missed_total}</div>
            <div class="neb-kpi-label">Missed Detected</div>
        </div>
        <div class="neb-kpi sun flare-animate-in flare-animate-in-delay-2">
            <div class="neb-kpi-icon">🤖</div>
            <div class="neb-kpi-value">{auto_reply_count}</div>
            <div class="neb-kpi-label">Auto Follow-Ups</div>
        </div>
        <div class="neb-kpi teal flare-animate-in flare-animate-in-delay-3">
            <div class="neb-kpi-icon">📈</div>
            <div class="neb-kpi-value">{recovery_rate:.0f}%</div>
            <div class="neb-kpi-label">Recovery Rate</div>
        </div>
        <div class="neb-kpi rose flare-animate-in flare-animate-in-delay-4">
            <div class="neb-kpi-icon">💾</div>
            <div class="neb-kpi-value">{leads_saved}</div>
            <div class="neb-kpi-label">Leads Saved</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # This Week vs Last Week
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 📅 This Week vs Last Week")

    week_col1, week_col2 = st.columns(2)
    with week_col1:
        delta_w = missed_this_week - missed_last_week
        delta_color = "normal" if delta_w <= 0 else "inverse"
        st.metric(label="Missed This Week", value=missed_this_week,
                  delta=f"{delta_w:+d} vs last week" if missed_last_week > 0 else None,
                  delta_color=delta_color)
    with week_col2:
        st.metric(label="Missed Last Week", value=missed_last_week)

    if HAS_PLOTLY and (missed_this_week > 0 or missed_last_week > 0):
        fig = go.Figure(data=[
            go.Bar(name='Last Week', x=['Missed Leads'], y=[missed_last_week],
                   marker_color='rgba(107, 122, 144, 0.5)'),
            go.Bar(name='This Week', x=['Missed Leads'], y=[missed_this_week],
                   marker_color='#ff6b6b'),
        ])
        fig.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#6b7a90', showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=20, b=40),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,212,255,0.06)'),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # 4-Week Trend
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 📈 4-Week Trend")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-0.5rem;'>Missed leads over the past 4 weeks to spot patterns.</p>", unsafe_allow_html=True)

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
                    mode="lines+markers", name="Total Leads",
                    line=dict(color="#00d4ff", width=2), marker=dict(size=8),
                ))
                fig.add_trace(go.Scatter(
                    x=weekly["week"], y=weekly["missed"],
                    mode="lines+markers", name="Missed Leads",
                    line=dict(color="#ff6b6b", width=2), marker=dict(size=8),
                    fill="tozeroy", fillcolor="rgba(255,107,107,0.06)",
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#6b7a90', showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=40, r=40, t=20, b=40),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,212,255,0.06)', title='Leads'),
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
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 💡 Estimated Leads Saved")

    saved_col1, saved_col2 = st.columns(2)
    with saved_col1:
        st.markdown(f"""
        <div style="background: var(--ember-dim); border:1px solid rgba(255,94,58,0.15); border-radius:var(--radius-md); padding:1.25rem; text-align:center;">
            <div style="font-size:0.72rem; color:var(--ember); text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:0.5rem;">Without System</div>
            <div style="font-size:2.5rem; font-weight:700; color:var(--ember); font-family:'JetBrains Mono',monospace;">{missed_total}</div>
            <div style="font-size:0.82rem; color:var(--text-secondary); margin-top:0.25rem;">leads would have been lost</div>
        </div>
        """, unsafe_allow_html=True)
    with saved_col2:
        st.markdown(f"""
        <div style="background: var(--teal-dim); border:1px solid rgba(45,212,191,0.15); border-radius:var(--radius-md); padding:1.25rem; text-align:center;">
            <div style="font-size:0.72rem; color:var(--teal); text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:0.5rem;">With System</div>
            <div style="font-size:2.5rem; font-weight:700; color:var(--teal); font-family:'JetBrains Mono',monospace;">{leads_saved}</div>
            <div style="font-size:0.82rem; color:var(--text-secondary); margin-top:0.25rem;">leads recovered via auto-reply</div>
        </div>
        """, unsafe_allow_html=True)

    if HAS_PLOTLY and missed_total > 0:
        fig = go.Figure(data=[
            go.Bar(name='Without System', x=['Leads'], y=[missed_total], marker_color='#ff6b6b'),
            go.Bar(name='Recovered', x=['Leads'], y=[leads_saved], marker_color='#10b981'),
        ])
        fig.update_layout(
            barmode='group', title='Leads Lost vs Leads Recovered',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#6b7a90', showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,212,255,0.06)'),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif missed_total == 0:
        st.info("No missed leads yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Recovery Details
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Recovery Breakdown")

    if missed_total > 0:
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        with detail_col1:
            auto_pct = (auto_reply_count / missed_total * 100) if missed_total > 0 else 0
            st.metric(label="Auto-Replied", value=auto_reply_count, delta=f"{auto_pct:.0f}% of missed")
        with detail_col2:
            human_pct = (recovered / missed_total * 100) if missed_total > 0 else 0
            st.metric(label="Human Follow-Up", value=recovered, delta=f"{human_pct:.0f}% of missed")
        with detail_col3:
            pending = max(0, missed_total - auto_reply_count - recovered)
            st.metric(label="Still Pending", value=pending,
                      delta="Needs attention" if pending > 0 else "All handled",
                      delta_color="inverse" if pending > 0 else "normal")

        if HAS_PLOTLY:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=recovery_rate,
                title={"text": "Recovery Rate (%)", "font": {"color": "#e8ecf1"}},
                number={"suffix": "%", "font": {"color": "#e8ecf1"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#6b7a90"},
                    "bar": {"color": "#10b981"},
                    "bgcolor": "rgba(17,13,26,0.5)",
                    "steps": [
                        {"range": [0, 50], "color": "rgba(255,107,107,0.12)"},
                        {"range": [50, 80], "color": "rgba(251,191,36,0.12)"},
                        {"range": [80, 100], "color": "rgba(16,185,129,0.12)"},
                    ],
                    "threshold": {"line": {"color": "#00d4ff", "width": 3}, "thickness": 0.8, "value": recovery_rate},
                },
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#6b7a90', height=300,
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
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Workflow Configuration & Business Settings")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-0.5rem;'>Update templates, courses, placement rates, and notification boundaries. Changes save immediately.</p>", unsafe_allow_html=True)

    save_status = st.empty()

    with st.form("settings_form"):
        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("##### 🏢 Business Identity")
            company_name = st.text_input("Company Name", overrides.get("COMPANY_NAME", config.COMPANY_NAME))
            sender_name = st.text_input("Sender Name", overrides.get("SENDER_NAME", config.SENDER_NAME))
            team_phone = st.text_input("Contact Phone", overrides.get("TEAM_PHONE", config.TEAM_PHONE))
            team_email = st.text_input("Contact Email", overrides.get("TEAM_EMAIL", config.TEAM_EMAIL))
            website_url = st.text_input("Website URL", overrides.get("WEBSITE_URL", config.WEBSITE_URL))

            st.markdown("##### ⚠️ Escalation Rules")
            overdue_hrs = st.number_input("Overdue Alert (Hours)", value=int(overrides.get("HOURS_BEFORE_OVERDUE", config.HOURS_BEFORE_OVERDUE)), min_value=1)
            esc_hrs = st.number_input("Escalation (Hours)", value=int(overrides.get("HOURS_BEFORE_ESCALATION", config.HOURS_BEFORE_ESCALATION)), min_value=1)

            st.markdown("##### 📊 Recovery Rate Alert")
            recovery_threshold = st.slider(
                "Alert when recovery rate drops below (%)", min_value=10, max_value=100,
                value=int(overrides.get("RECOVERY_RATE_THRESHOLD", config.RECOVERY_RATE_THRESHOLD)),
                help="Email alert is sent when fewer than this % of missed leads are handled.",
            )

        with sc2:
            st.markdown("##### 📚 Course Offerings")
            placement_rate = st.text_input("Placement Rate", overrides.get("PLACEMENT_RATE", config.PLACEMENT_RATE))
            partners = st.text_input("Hiring Partners", overrides.get("COMPANY_PARTNERS", config.COMPANY_PARTNERS))
            discount = st.text_area("Discount Text", overrides.get("DISCOUNT_INFO", config.DISCOUNT_INFO), height=80)
            scholarship = st.text_area("Scholarship Info", overrides.get("SCHOLARSHIP_INFO", config.SCHOLARSHIP_INFO), height=80)
            emi_text = st.text_area("EMI Details", overrides.get("EMI_INFO", config.EMI_INFO), height=80)

            st.markdown("##### ⏱️ Reply Delay (Seconds)")
            td1, td2 = st.columns(2)
            with td1:
                min_delay = st.number_input("Min Delay", value=int(overrides.get("MIN_REPLY_DELAY", config.MIN_REPLY_DELAY)), min_value=1)
            with td2:
                max_delay = st.number_input("Max Delay", value=int(overrides.get("MAX_REPLY_DELAY", config.MAX_REPLY_DELAY)), min_value=1)

        st.markdown("##### ✍️ Email Signature")
        sig_val = overrides.get("EMAIL_SIGNATURE", f"Best regards,\n{sender_name}\n{company_name}\nPhone: {team_phone}\nEmail: {team_email}\nWeb: {website_url}")
        email_signature = st.text_area("Email Signature Block", sig_val, height=120)

        submitted = st.form_submit_button("💾 Save & Update Settings", type="primary")

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
            save_status.success("✅ Settings saved successfully!")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Environment Variables Panel
    st.markdown("<div class='neb-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔒 Environment Variables")

    env_list = {
        "SMTP_USER": _get_secret("SMTP_USER"),
        "SMTP_HOST": "smtp.gmail.com",
        "IMAP_USER": _get_secret("IMAP_USER"),
        "NOTIFY_EMAIL": _get_secret("NOTIFY_EMAIL"),
        "SENDER_NAME": _get_secret("SENDER_NAME"),
    }

    for key, val in env_list.items():
        val_str = "❌ Not Set" if not val else f"✅ {val}"
        st.markdown(f"**{key}:** `{val_str}`")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="neb-footer">
    Missed-Lead Detector &bull; AI-Powered Sales Command Center &bull; CIT Chennai
</div>
""", unsafe_allow_html=True)
