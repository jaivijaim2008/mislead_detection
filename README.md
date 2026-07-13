# Missed-Lead Detector

Automated inbox monitoring, smart replies & follow-up management for sales teams.

## Overview

A production-ready system that monitors your Gmail inbox, detects missed leads using ML, automatically sends human-like replies, and alerts the sales team when follow-ups are overdue.

### Features

- **Smart Auto-Replies** — Intent-aware templates that read naturally
- **Live Gmail Integration** — IMAP fetch + SMTP threaded replies, filters newsletters/promotions
- **ML + Deep Learning** — 8 models including Optuna-tuned XGBoost (AUC 0.9794) and PyTorch neural network
- **Multi-Channel Notifications** — Email alerts + Streamlit dashboard
- **Follow-Up Tracking** — Flags leads needing human attention after 24h
- **Sales Team Dashboard** — Command center with leads, replies, and alerts (6 pages)
- **REST API** — FastAPI endpoints for scoring, reply preview, and lead listing
- **SQLite Database** — Production-ready storage replacing JSON files
- **163+ Unit Tests** — Comprehensive pytest suite with 93%+ coverage

## 🚀 Deploy to Streamlit Cloud (Fast & Free)

### Step 1: Push your code to GitHub

```bash
# If you haven't already
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/missed-lead-detector.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"** → select your repo
3. Set the following:
   - **Repository**: `YOUR_USERNAME/missed-lead-detector`
   - **Branch**: `main`
   - **Main file path**: `missed_lead_detector/src/dashboard.py`
4. Click **"Deploy"**

> ⏱️ **Cold start**: ~15-30 seconds on first load. After that, the app stays warm.
> 💸 **Cost**: Free (public repos get unlimited app hours; Streamlit Community Cloud)

### Step 3: Add Secrets

1. In your Streamlit Cloud dashboard, go to **Settings → Secrets**
2. Paste the following (with **your real credentials**):

```toml
# ── Dashboard Login ────────────────────────────────────
AUTH_USER = "admin"
AUTH_PASS = "your-strong-password"

# ── Gmail IMAP (read inbox) ────────────────────────────
IMAP_USER = "your-email@gmail.com"
IMAP_PASS = "your-16-char-app-password"

# ── Gmail SMTP (send replies) ──────────────────────────
SMTP_USER = "your-email@gmail.com"
SMTP_PASS = "your-16-char-app-password"

# ── Notification email ─────────────────────────────────
NOTIFY_EMAIL = "your-email@gmail.com"
SENDER_NAME = "Sales Team"
```

> 📧 **Gmail App Password**: Enable 2FA on your Google Account → Security → App Passwords → generate one for "Mail"

3. Click **"Save"** — the app will restart automatically with your secrets loaded

### Step 4: Access your app

Your app will be live at:
`https://YOUR_APP_NAME.streamlit.app`

Login with the `AUTH_USER` / `AUTH_PASS` you set in Secrets.

---

## 📦 Local Development

```bash
cd missed_lead_detector

# Create virtual env
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run src/dashboard.py

# Run the FastAPI server (optional)
uvicorn src.api:app --reload --port 8000
```

### Run Tests

```bash
# All 163+ tests
pytest tests/ -v

# Specific test files
pytest tests/test_database.py -v
pytest tests/test_api.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
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

## 🏗 Architecture

```
GMAIL INBOX (IMAP) → FILTER NEWSLETTERS → ML SCORING
        ↓
Smart Reply Engine (intent detection + templates)
        ↓
Auto-Reply via SMTP (threaded, human-like)
        ↓
Notifications (email + dashboard)
        ↓
Follow-Up Tracker (flags overdue leads after 24h)
```

## 🔌 API Endpoints

When running the FastAPI server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/score` | POST | Score a single lead |
| `/api/v1/score/batch` | POST | Score batch leads |
| `/api/v1/reply/preview` | POST | Preview smart reply |
| `/api/v1/reply/send` | POST | Send auto-reply |
| `/api/v1/leads` | GET | List scored leads |
| `/api/v1/leads/{id}` | GET | Get lead details |
| `/api/v1/leads/{id}/followup` | PUT | Mark followed up |
| `/api/v1/stats` | GET | Pipeline stats |
| `/api/v1/scan` | POST | Trigger inbox scan |

Docs available at `/docs` (Swagger) or `/redoc` (ReDoc).

## 📊 Model Performance

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

## 🛠 Tech Stack

- **Python 3.11** — Core language
- **Scikit-learn** — ML models & evaluation
- **XGBoost** — Gradient boosting (Optuna-tuned)
- **PyTorch** — Deep learning
- **Optuna** — Hyperparameter optimization
- **Streamlit** — Dashboard UI
- **FastAPI** — REST API
- **SQLite** — Database storage
- **Pytest** — Testing (163+ tests)
- **IMAP/SMTP** — Email integration
- **Fly.io** — Fast alternative deployment option

## 📁 Project Structure

```
missed_lead_detector/
├── src/
│   ├── dashboard.py           # Streamlit dashboard (entry point)
│   ├── api.py                 # FastAPI REST endpoints
│   ├── api_models.py          # Pydantic request/response models
│   ├── database.py            # SQLite storage backend
│   ├── orchestrator.py        # Pipeline runner
│   ├── train_model.py         # ML + DL training pipeline
│   ├── deep_learning.py       # PyTorch neural network
│   ├── xgb_tuning.py          # Optuna XGBoost tuning
│   ├── email_reader.py        # IMAP Gmail reader
│   ├── auto_followup.py       # SMTP follow-up sender
│   ├── inbox_monitor.py       # Main monitoring pipeline
│   ├── smart_reply_engine.py  # Intent detection + reply gen
│   ├── notifications.py       # Email/desktop notifications
│   ├── daily_digest.py        # Daily summary email
│   ├── employee_reminder.py   # Desktop popup alerts
│   ├── generate_data.py       # Synthetic data generator
│   ├── merge_kaggle_data.py   # Kaggle dataset merger
│   ├── config.py              # Business configuration
│   ├── hf_intent_model.py     # HuggingFace intent model
│   ├── mcnemar_test.py        # Statistical significance test
│   └── plot_model_comparison.py  # Model comparison charts
├── tests/                     # 19 test files (163+ tests)
├── tools/                     # Utility scripts
├── data/                      # Training data
├── models/                    # Trained model artifacts
├── outputs/                   # Charts, reports, scored data
├── logs/                      # Pipeline run logs
├── requirements.txt           # Python dependencies
├── packages.txt               # System-level apt packages
├── .streamlit/
│   ├── config.toml            # Streamlit server config
│   └── secrets.toml           # Secrets template
├── Dockerfile                 # Docker deployment
├── fly.toml                   # Fly.io deployment
├── Procfile                   # Heroku deployment
└── .github/workflows/test.yml # CI pipeline
```

## 📜 License

Academic project — CIT Chennai, Batch 2025-27
