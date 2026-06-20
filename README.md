# Missed-Lead Detector

AI-Powered Missed-Lead Detection & Automated Follow-Up System.

## Live Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-app-url.streamlit.app)

## Overview

A machine learning system that automatically identifies "missed leads" — customers who expressed interest but did not receive a timely response — and triggers automated follow-up emails.

### Features

- **Live Gmail Integration** — Fetches real emails via IMAP, filters newsletters/promotions
- **ML + Deep Learning** — 8 models including Optuna-tuned XGBoost (AUC 0.9794) and PyTorch neural network
- **Auto Follow-Up** — Sends threaded follow-up emails via SMTP
- **Interactive Dashboard** — Streamlit UI with real-time pipeline controls
- **Kaggle Data** — Trained on 13,740 real-world samples (X Education + Customer Support + Synthetic)

## Quick Start

### Streamlit Cloud (Recommended)

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file path: `missed_lead_detector/src/dashboard.py`
5. Click "Deploy"

### Local

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

## Architecture

```
GMAIL INBOX (IMAP) → FILTER NEWSLETTERS → FEATURE ENGINEERING
        ↓
ML PIPELINE (LR, NB, DT, RF, XGBoost, Ensemble, K-Means)
        ↓
DEEP LEARNING (PyTorch GPU — ResNet with BatchNorm + Residual Blocks)
        ↓
GRAND ENSEMBLE (50% ML + 50% DL)
        ↓
INFERENCE: missed_probability ≥ 0.50 → MISSED LEAD
        ↓                        ↓
auto_followup.py      employee_reminder.py
(SMTP threaded email)  (repeating popup)
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
