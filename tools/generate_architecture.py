"""
generate_architecture.py — Missed-Lead Detector
Generates a professional system architecture diagram.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# ── Figure Setup ──────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0a0a12')
ax.set_facecolor('#0a0a12')

# ── Color Palette ─────────────────────────────────────────
BLUE      = '#38bdf8'
GREEN     = '#34d399'
RED       = '#f87171'
AMBER     = '#fbbf24'
PURPLE    = '#a78bfa'
CYAN      = '#22d3ee'
WHITE     = '#e6edf3'
GRAY      = '#7d8590'
DARK_BG   = '#141822'
DARKER_BG = '#0d1117'

def draw_box(ax, x, y, w, h, label, color, fontsize=10, sublabel=None, icon=None):
    """Draw a rounded rectangle box with label."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=DARK_BG, edgecolor=color,
                         linewidth=2, zorder=3)
    ax.add_patch(box)

    # Glow effect
    glow = FancyBboxPatch((x - 0.05, y - 0.05), w + 0.1, h + 0.1,
                          boxstyle="round,pad=0.2",
                          facecolor='none', edgecolor=color,
                          linewidth=0.5, alpha=0.3, zorder=2)
    ax.add_patch(glow)

    display_label = f"{icon} {label}" if icon else label
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), display_label,
            ha='center', va='center', fontsize=fontsize,
            color=WHITE, fontweight='bold', zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel,
                ha='center', va='center', fontsize=8,
                color=GRAY, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color=GRAY, style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                               lw=lw, connectionstyle='arc3,rad=0'),
                zorder=5)

def draw_curved_arrow(ax, x1, y1, x2, y2, color=GRAY, rad=0.2, lw=1.5):
    """Draw a curved arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                               lw=lw, connectionstyle=f'arc3,rad={rad}'),
                zorder=5)

# ── Title ─────────────────────────────────────────────────
ax.text(9, 9.5, 'Missed-Lead Detector — System Architecture',
        ha='center', va='center', fontsize=20, color=WHITE,
        fontweight='bold')
ax.text(9, 9.1, 'AI-Powered Email Monitoring • Smart Auto-Replies • Sales Pipeline Retention',
        ha='center', va='center', fontsize=11, color=GRAY)

# ── Layer 1: Data Sources ─────────────────────────────────
ax.text(2.5, 8.3, 'DATA SOURCES', ha='center', fontsize=9, color=GRAY, fontweight='bold')

draw_box(ax, 0.3, 7.3, 2.2, 0.8, 'Gmail Inbox', BLUE, fontsize=10, sublabel='IMAP4_SSL', icon='📧')
draw_box(ax, 2.8, 7.3, 2.2, 0.8, 'Synthetic Data', GREEN, fontsize=10, sublabel='500 leads', icon='📊')
draw_box(ax, 5.3, 7.3, 2.2, 0.8, 'Kaggle Datasets', PURPLE, fontsize=10, sublabel='13,240 leads', icon='📦')

# ── Layer 2: Processing Pipeline ──────────────────────────
ax.text(9, 6.5, 'PROCESSING PIPELINE', ha='center', fontsize=9, color=GRAY, fontweight='bold')

draw_box(ax, 0.3, 5.4, 2.0, 0.8, 'Newsletter Filter', GRAY, fontsize=9, sublabel='Promo detection', icon='🔍')
draw_box(ax, 2.6, 5.4, 2.0, 0.8, 'Feature Engineering', AMBER, fontsize=9, sublabel='9 features', icon='⚙️')
draw_box(ax, 4.9, 5.4, 2.2, 0.8, 'ML Scoring', GREEN, fontsize=9, sublabel='8 models + Ensemble', icon='🤖')
draw_box(ax, 7.4, 5.4, 2.2, 0.8, 'DL Scoring', PURPLE, fontsize=9, sublabel='PyTorch NN', icon='🧠')
draw_box(ax, 9.9, 5.4, 2.0, 0.8, 'Grand Ensemble', RED, fontsize=9, sublabel='ML+DL 50/50', icon='🎯')

# ── Layer 3: Action Layer ─────────────────────────────────
ax.text(14, 6.5, 'ACTION LAYER', ha='center', fontsize=9, color=GRAY, fontweight='bold')

draw_box(ax, 12.2, 5.4, 2.0, 0.8, 'Smart Reply', CYAN, fontsize=9, sublabel='8 intents', icon='💬')
draw_box(ax, 14.5, 5.4, 2.0, 0.8, 'Auto-Followup', AMBER, fontsize=9, sublabel='SMTP send', icon='📤')
draw_box(ax, 12.2, 4.2, 2.0, 0.8, 'Notifications', RED, fontsize=9, sublabel='Multi-channel', icon='🔔')
draw_box(ax, 14.5, 4.2, 2.0, 0.8, 'Follow-Up Track', GREEN, fontsize=9, sublabel='24h threshold', icon='⏱️')

# ── Layer 4: Output / UI ──────────────────────────────────
ax.text(9, 3.4, 'OUTPUT & MONITORING', ha='center', fontsize=9, color=GRAY, fontweight='bold')

draw_box(ax, 0.3, 2.2, 3.0, 1.0, 'Streamlit Dashboard', BLUE, fontsize=11, sublabel='6 pages • Aurora theme', icon='📊')
draw_box(ax, 3.6, 2.2, 2.8, 1.0, 'Email Alerts', AMBER, fontsize=10, sublabel='Sales team alerts', icon='📧')
draw_box(ax, 6.7, 2.2, 2.8, 1.0, 'Desktop Popups', RED, fontsize=10, sublabel='tkinter alerts', icon='🖥️')
draw_box(ax, 9.8, 2.2, 3.0, 1.0, 'Daily Digest Report', GREEN, fontsize=10, sublabel='HTML email report', icon='📋')

# ── Layer 5: Infrastructure ───────────────────────────────
ax.text(9, 1.3, 'INFRASTRUCTURE', ha='center', fontsize=9, color=GRAY, fontweight='bold')

draw_box(ax, 0.3, 0.2, 2.5, 0.8, 'GitHub Actions', GRAY, fontsize=9, sublabel='10-min schedule', icon='⏰')
draw_box(ax, 3.1, 0.2, 2.5, 0.8, 'Streamlit Cloud', BLUE, fontsize=9, sublabel='Auto-deploy', icon='☁️')
draw_box(ax, 5.9, 0.2, 2.5, 0.8, 'Optuna Tuning', PURPLE, fontsize=9, sublabel='100 trials', icon='⚙️')
draw_box(ax, 8.7, 0.2, 2.5, 0.8, 'Model Artifacts', GREEN, fontsize=9, sublabel='pkl + pt files', icon='💾')
draw_box(ax, 11.5, 0.2, 2.5, 0.8, 'Log Files', AMBER, fontsize=9, sublabel='JSON tracking', icon='📝')

# ── Arrows: Data Sources → Processing ─────────────────────
draw_arrow(ax, 1.4, 7.3, 1.3, 6.2, color=BLUE)
draw_arrow(ax, 3.9, 7.3, 3.6, 6.2, color=GREEN)
draw_arrow(ax, 6.4, 7.3, 3.6, 6.2, color=PURPLE)

# ── Arrows: Processing Pipeline flow ──────────────────────
draw_arrow(ax, 2.3, 5.8, 2.6, 5.8, color=GRAY)
draw_arrow(ax, 4.6, 5.8, 4.9, 5.8, color=GRAY)
draw_arrow(ax, 7.1, 5.8, 7.4, 5.8, color=GREEN)
draw_arrow(ax, 9.6, 5.8, 9.9, 5.8, color=PURPLE)

# ── Arrows: Processing → Action ───────────────────────────
draw_arrow(ax, 11.9, 5.8, 12.2, 5.8, color=RED)
draw_arrow(ax, 11.9, 5.6, 12.2, 4.6, color=RED)

# ── Arrows: Action internal ───────────────────────────────
draw_arrow(ax, 14.2, 5.8, 14.5, 5.8, color=AMBER)
draw_arrow(ax, 13.2, 5.4, 13.2, 5.0, color=CYAN, lw=1)

# ── Arrows: Action → Output ───────────────────────────────
draw_curved_arrow(ax, 13.2, 4.2, 3.0, 3.2, color=RED, rad=-0.3)
draw_curved_arrow(ax, 15.5, 4.2, 5.0, 3.2, color=GREEN, rad=-0.2)
draw_curved_arrow(ax, 15.5, 5.4, 8.0, 3.2, color=AMBER, rad=-0.1)

# ── Arrows: Output → Infrastructure ───────────────────────
draw_arrow(ax, 1.8, 2.2, 1.5, 1.0, color=BLUE, lw=1)
draw_arrow(ax, 4.5, 2.2, 4.3, 1.0, color=AMBER, lw=1)

# ── Legend ─────────────────────────────────────────────────
legend_items = [
    (BLUE, 'Email Integration'),
    (GREEN, 'ML Models'),
    (PURPLE, 'Deep Learning'),
    (RED, 'Notifications'),
    (AMBER, 'Auto-Reply'),
    (CYAN, 'Smart Engine'),
]

for i, (color, label) in enumerate(legend_items):
    x = 13.0 + (i % 3) * 1.7
    y = 1.3 - (i // 3) * 0.4
    circle = plt.Circle((x, y), 0.12, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(x + 0.2, y, label, fontsize=7, color=GRAY, va='center')

# ── Save ──────────────────────────────────────────────────
output_path = os.path.join(OUT, "system_architecture.png")
plt.savefig(output_path, dpi=200, bbox_inches='tight',
            facecolor='#0a0a12', edgecolor='none')
plt.close()
print(f"[architecture] Saved system architecture diagram -> {output_path}")
