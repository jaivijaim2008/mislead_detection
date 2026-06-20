# Missed-Lead Detector

Automated inbox monitoring, smart replies & follow-up management for sales teams.

## Live Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-app-url.streamlit.app)

## Overview

A production-ready system that monitors your Gmail inbox, detects missed leads using ML, automatically sends human-like replies (clients cannot tell it is automated), and alerts the sales team when follow-ups are overdue.

### How It Works

1. **Inbox Monitor** scans Gmail every 10 minutes (via GitHub Actions)
2. **ML Model** scores each email for missed-lead probability (AUC 0.9794)
3. **Smart Reply Engine** detects intent and generates a contextual, human-like reply
4. **Auto-Reply** is sent via SMTP — the client sees a normal manual response
5. **Notifications** alert the sales team via email + dashboard + desktop popups
6. **Follow-Up Tracker** flags leads needing human attention after 24 hours

### Features

- **Smart Auto-Replies** — Intent-aware templates that read naturally; clients cannot tell they are automated
- **Live Gmail Integration** — IMAP fetch + SMTP threaded replies, filters newsletters/promotions
- **ML + Deep Learning** — 8 models including Optuna-tuned XGBoost (AUC 0.9794) and PyTorch neural network
- **Multi-Channel Notifications** — Email alerts + Streamlit dashboard + desktop popups
- **Follow-Up Tracking** — Auto-detects when sales team has not followed up within 24h
- **Sales Team Dashboard** — Clean command center showing all leads, replies, and alerts
- **GitHub Actions Scheduling** — Runs automatically every 10 minutes

## Quick Start

### Streamlit Cloud (Recommended)

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file path: `missed_lead_detector/src/dashboard.py`
5. Set secrets (SMTP_USER, SMTP_PASS, IMAP_USER, IMAP_PASS)
6. Click "Deploy"

### Local

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

### Run Inbox Monitor

```bash
# Single scan
python src/inbox_monitor.py

# Continuous monitoring (every 5 min)
python src/inbox_monitor.py --loop 300

# Dry run (no emails sent)
python src/inbox_monitor.py --dry-run
```

## Architecture

```
GMAIL INBOX (IMAP) → FILTER NEWSLETTERS → ML SCORING
        ↓
Smart Reply Engine (intent detection + templates)
        ↓
Auto-Reply via SMTP (threaded, human-like)
        ↓
Notifications (email + dashboard + desktop)
        ↓
Follow-Up Tracker (flags overdue leads after 24h)
```

## Model Performance

| Model | Test AUC |
|-------|----------|
| XGBoost (Optuna-tuned) | **0.9794** |
| RandomForest | 0.9725 |
| Ensemble (RF+XGB+LR) | 0.9720 |
| Grand Ensemble (ML+DL) | 0.9709 |
| Deep Learning (PyTorch) | 0.9613 |
| Decision Tree | 0.9627 |
| Logistic Regression | 0.9132 |
| Naive Bayes | 0.8604 |

## Tech Stack

- **Python 3.11** — Core language
- **Scikit-learn** — ML models & evaluation
- **XGBoost** — Gradient boosting (Optuna-tuned)
- **PyTorch** — Deep learning (GPU-accelerated)
- **Optuna** — Hyperparameter optimization
- **Streamlit** — Dashboard UI
- **IMAP/SMTP** — Email integration

## Dashboard Pages

| Page | Description |
|------|-------------|
| Live Dashboard | Pipeline controls, inbox metrics, scored emails |
| Inbox Leads | Filterable/sortable leads with customer details |
| Model Results | 8-model comparison, tuning chart, DL charts |
| Sent Follow-Ups | Email dedup log |
| EDA | Interactive visualizations |
| About | Architecture, tech stack, SDG alignment |

## Project Structure

```
missed_lead_detector/
├── src/
│   ├── dashboard.py           # Streamlit dashboard
│   ├── orchestrator.py        # Pipeline runner (demo/live/preview)
│   ├── train_model.py         # ML + DL training pipeline
│   ├── deep_learning.py       # PyTorch neural network
│   ├── xgb_tuning.py          # Optuna XGBoost tuning
│   ├── email_reader.py        # IMAP Gmail reader
│   ├── auto_followup.py       # SMTP follow-up sender
│   ├── employee_reminder.py   # Desktop popup alerts
│   ├── generate_data.py       # Synthetic data generator
│   ├── merge_kaggle_data.py   # Kaggle dataset merger
│   └── plot_model_comparison.py  # AUC comparison chart
├── data/
│   ├── leads.csv              # Synthetic training data
│   └── leads_merged.csv       # Merged (synthetic + Kaggle)
├── models/                    # Trained model artifacts
├── outputs/                   # Charts, reports, scored data
├── notebooks/                 # EDA notebook
├── logs/                      # Pipeline run logs
├── requirements.txt           # Cloud deployment deps
├── .streamlit/config.toml     # Streamlit config
└── .gitignore
```

## License

Academic project — CIT Chennai, Batch 2025-27
