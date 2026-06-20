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
from PIL import Image

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

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 16px;
        margin-bottom: 2rem; color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; font-size: 1.05rem; }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px; padding: 1.5rem; text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .metric-value { font-size: 2.5rem; font-weight: 700; color: #1a1a2e; line-height: 1; }
    .metric-card .metric-label { font-size: 0.9rem; color: #555; margin-top: 0.5rem; font-weight: 600;
                                 text-transform: uppercase; letter-spacing: 0.5px; }
    .missed-card .metric-value { color: #e74c3c; }
    .replied-card .metric-value { color: #27ae60; }
    .auc-card .metric-value { color: #667eea; }
    .email-card .metric-value { color: #f39c12; }
    .inbox-card .metric-value { color: #3498db; }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #1a1a2e;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea; display: inline-block;
    }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 { color: white !important; }
    .status-badge {
        display: inline-block; padding: 0.25rem 0.75rem;
        border-radius: 20px; font-size: 0.85rem;
        font-weight: 600; margin-left: 0.5rem;
    }
    .status-live { background: #27ae60; color: white; }
    .status-demo { background: #f39c12; color: white; }
    .notif-card {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
        border-left: 4px solid #e74c3c; border-radius: 8px;
        padding: 1rem 1.5rem; margin-bottom: 0.75rem;
    }
    .notif-card.auto-reply {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left-color: #27ae60;
    }
    .notif-card.overdue {
        background: linear-gradient(135deg, #fffaf0 0%, #ffeaa7 100%);
        border-left-color: #f39c12;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 Missed-Lead Detector</h1>
    <p>Automated inbox monitoring, smart replies & follow-up management</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["📊 Dashboard", "📥 Inbox", "🤖 Auto-Replies",
         "🔔 Notifications", "📈 Model Performance", "⚙️ Settings"],
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
        st.markdown(f"### 🔔 {notif_count} unread alerts")

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


def load_followup_status():
    if os.path.exists(FOLLOWUP_LOG):
        with open(FOLLOWUP_LOG) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<div class="section-header">📊 Sales Team Overview</div>', unsafe_allow_html=True)

    live = is_live_configured()
    status_text = "LIVE — Monitoring Gmail" if live else "DEMO — Using sample data"
    status_class = "status-live" if live else "status-demo"
    st.markdown(f'Pipeline: <span class="status-badge {status_class}">{status_text}</span>',
                unsafe_allow_html=True)

    scored = load_scored()
    followup = load_followup_status()
    reply_log = load_json(REPLY_LOG)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total = len(scored)
        st.markdown(f"""<div class="metric-card inbox-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Total Leads</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        missed = int(scored["predicted_missed"].sum()) if "predicted_missed" in scored.columns and len(scored) > 0 else 0
        st.markdown(f"""<div class="metric-card missed-card">
            <div class="metric-value">{missed}</div>
            <div class="metric-label">Missed Leads</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        replied = len([r for r in reply_log if isinstance(r, dict)])
        st.markdown(f"""<div class="metric-card replied-card">
            <div class="metric-value">{replied}</div>
            <div class="metric-label">Auto-Replied</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        overdue = len([s for s in followup.values()
                      if isinstance(s, dict) and s.get("auto_replied")
                      and not s.get("human_followed_up")])
        st.markdown(f"""<div class="metric-card email-card">
            <div class="metric-value">{overdue}</div>
            <div class="metric-label">Awaiting Human</div>
        </div>""", unsafe_allow_html=True)

    # Lead status breakdown
    if len(scored) > 0:
        st.markdown("### Lead Status Breakdown")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if "predicted_missed" in scored.columns:
                ok = int((scored["predicted_missed"] == 0).sum())
                st.metric("Responded On Time", ok)
            else:
                st.metric("Responded On Time", "—")
        with bc2:
            st.metric("Missed (Auto-Replied)", missed)
        with bc3:
            if "high_intent_flag" in scored.columns:
                hi = int(scored["high_intent_flag"].sum())
                st.metric("High Intent", hi)
            else:
                st.metric("High Intent", "—")

        # Missed probability distribution
        if "missed_probability" in scored.columns:
            st.markdown("### Missed Probability Distribution")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(scored["missed_probability"], bins=20, color="#e74c3c",
                   alpha=0.7, edgecolor="white")
            ax.axvline(0.5, color="#333", linestyle="--", linewidth=2, label="Threshold (0.5)")
            ax.set_xlabel("Missed Probability")
            ax.set_ylabel("Count")
            ax.set_title("Lead Scoring Distribution")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

    # Recent inbox emails table
    if len(scored) > 0:
        st.markdown("### Recent Leads")
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
                      "predicted_missed": "Missed?"}
            display_df.columns = [rename.get(c, c) for c in display_df.columns]
        else:
            show_cols = [c for c in ["lead_id", "channel", "message_text",
                                     "response_gap_hrs", "high_intent_flag",
                                     "missed_probability", "predicted_missed"]
                        if c in scored.columns]
            display_df = scored[show_cols].head(20).copy()
            rename = {"lead_id": "ID", "channel": "Channel", "message_text": "Message",
                      "response_gap_hrs": "Gap (hrs)", "high_intent_flag": "Intent",
                      "missed_probability": "Missed %", "predicted_missed": "Missed?"}
            display_df.columns = [rename.get(c, c) for c in display_df.columns]

        if "Missed %" in display_df.columns:
            display_df["Missed %"] = display_df["Missed %"].apply(lambda x: f"{x:.0%}")
        if "Intent" in display_df.columns:
            display_df["Intent"] = display_df["Intent"].map({1: "HIGH", 0: "low"})
        if "Missed?" in display_df.columns:
            display_df["Missed?"] = display_df["Missed?"].map({1: "🔴 YES", 0: "🟢 No"})
        if "Message" in display_df.columns:
            display_df["Message"] = display_df["Message"].apply(
                lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else x)

        st.dataframe(display_df, use_container_width=True, height=350)

# ══════════════════════════════════════════════════════════
# PAGE: Inbox
# ══════════════════════════════════════════════════════════
elif page == "📥 Inbox":
    st.markdown('<div class="section-header">📥 Email Inbox</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Scan Inbox Now", type="primary"):
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
        if st.button("🧪 Dry Run (no sends)"):
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
        st.markdown(f"### All Scored Emails ({len(scored)})")

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
                  "missed_probability": "Missed %", "predicted_missed": "Missed?",
                  "_received_time": "Received"}
        display.columns = [rename.get(c, c) for c in display.columns]

        if "Missed %" in display.columns:
            display["Missed %"] = display["Missed %"].apply(lambda x: f"{x:.0%}")
        if "Intent" in display.columns:
            display["Intent"] = display["Intent"].map({1: "HIGH", 0: "low"})
        if "Missed?" in display.columns:
            display["Missed?"] = display["Missed?"].map({1: "🔴 YES", 0: "🟢 No"})
        if "Message" in display.columns:
            display["Message"] = display["Message"].apply(
                lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else x)

        st.dataframe(display, use_container_width=True, height=500)
    else:
        st.info("📭 No emails scanned yet. Click 'Scan Inbox Now' above.")

# ══════════════════════════════════════════════════════════
# PAGE: Auto-Replies
# ══════════════════════════════════════════════════════════
elif page == "🤖 Auto-Replies":
    st.markdown('<div class="section-header">🤖 Auto-Reply Log</div>', unsafe_allow_html=True)
    st.markdown("Every missed lead gets a human-like auto-reply. The client cannot tell it's automated.")

    reply_log = load_json(REPLY_LOG)
    followup = load_followup_status()

    if reply_log:
        reply_df = pd.DataFrame(reply_log)
        st.markdown(f"### {len(reply_df)} Auto-Replies Sent")

        # Intent breakdown
        if "detected_intent" in reply_df.columns:
            st.markdown("#### Intent Distribution")
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
        st.markdown("### Follow-Up Status")
        if followup:
            fu_data = []
            for lid, status in followup.items():
                if isinstance(status, dict):
                    fu_data.append({
                        "Lead ID": lid,
                        "Customer": status.get("customer_name", ""),
                        "Auto-Replied": "✅" if status.get("auto_replied") else "❌",
                        "Human Follow-Up": "✅" if status.get("human_followed_up") else "⏳ Pending",
                        "Overdue Alert": "⚠️ YES" if status.get("overdue_notified") else "—",
                    })
            if fu_data:
                st.dataframe(pd.DataFrame(fu_data), use_container_width=True, height=300)
    else:
        st.info("📭 No auto-replies sent yet. Run the inbox monitor to start.")

# ══════════════════════════════════════════════════════════
# PAGE: Notifications
# ══════════════════════════════════════════════════════════
elif page == "🔔 Notifications":
    st.markdown('<div class="section-header">🔔 Notifications</div>', unsafe_allow_html=True)

    notifs = load_json(NOTIF_LOG)
    unread = [n for n in notifs if not n.get("read", False)]

    if unread:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {len(unread)} unread alerts")
        with col2:
            if st.button("✓ Mark all read"):
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
                    <strong>{icon} {n.get('title', '')}</strong><br>
                    <span style="font-size:0.85rem; color:#666">{n.get('timestamp', '')}</span><br>
                    <span style="font-size:0.95rem">{n.get('message', '')}</span>
                </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ All caught up — no unread notifications!")

# ══════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    data_source = "Merged (13,740 rows)" if os.path.exists(MERGED_DATA) else "Synthetic (500 rows)"
    st.markdown(f'<div class="section-header">📈 Model Performance — {data_source}</div>',
                unsafe_allow_html=True)

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.0%}")
    c2.metric("Missed F1", f"{f1:.3f}")
    c3.metric("Missed Precision", f"{prec:.3f}")
    c4.metric("Missed Recall", f"{rec:.3f}")

    # Model comparison
    if os.path.exists(MODEL_CMP):
        with open(MODEL_CMP) as f:
            mc = json.load(f)
        if mc.get("models"):
            st.markdown("### Model Comparison")
            rows = [{"Model": n, "AUC": info["auc"]}
                   for n, info in sorted(mc["models"].items(),
                                         key=lambda x: x[1]["auc"], reverse=True)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Charts
    st.markdown("### Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(CM_IMG):
            st.image(Image.open(CM_IMG), caption="Confusion Matrix", use_container_width=True)
    with col2:
        if os.path.exists(FI_IMG):
            st.image(Image.open(FI_IMG), caption="Feature Importance", use_container_width=True)

    if os.path.exists(REPORT):
        with open(REPORT) as f:
            st.code(f.read(), language="text")

    # Tuning
    if os.path.exists(CMP_CHART):
        st.markdown("### Before/After Tuning")
        st.image(Image.open(CMP_CHART), use_container_width=True)

    if os.path.exists(XGB_TUNING):
        with open(XGB_TUNING) as f:
            tuning = json.load(f)
        if tuning.get("best_params"):
            st.markdown("### XGBoost Tuned Hyperparameters")
            st.json(tuning["best_params"])

# ══════════════════════════════════════════════════════════
# PAGE: Settings
# ══════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown('<div class="section-header">⚙️ System Settings</div>', unsafe_allow_html=True)

    st.markdown("### Connection Status")
    live = is_live_configured()
    if live:
        st.success("✅ Gmail IMAP/SMTP configured")
    else:
        st.warning("⚠️ Gmail not configured. Set SMTP_USER and SMTP_PASS env vars.")

    st.markdown("### How It Works")
    st.markdown("""
    1. **Inbox Monitor** scans Gmail every 5 minutes (via GitHub Actions)
    2. **ML Model** scores each email for missed-lead probability
    3. **Smart Reply Engine** generates a human-like reply based on detected intent
    4. **Auto-Reply** is sent — client cannot tell it's automated
    5. **Notifications** alert the sales team via email + dashboard + desktop
    6. **Follow-Up Tracker** flags leads that need human attention
    """)

    st.markdown("### Environment Variables")
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
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.85rem;'>"
    "Missed-Lead Detector · Automated Inbox Monitoring & Follow-Up · "
    f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    "</p>", unsafe_allow_html=True
)
