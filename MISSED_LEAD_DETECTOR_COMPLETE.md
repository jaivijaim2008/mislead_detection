# Missed-Lead Detector — Complete Project Handover

> **AI-Powered Missed-Lead Detection and Automated Follow-Up System for Customer Retention in Sales Pipelines**
>
> Batch 2025-27 · II Year · III Semester · Section: AIDS
> Team Member 1 : **JAI VIJAI M** — 210425243091
> Team Member 2 : **ARUNKUMAR I** — 210425243028
> Email on record: jaivijaim.aids2025@citchennai.net

---

## Table of Contents

1. [Google Form — Exact Answers to Paste](#1-google-form--exact-answers-to-paste)
2. [Project Description](#2-project-description)
3. [System Architecture](#3-system-architecture)
4. [Course Unit → Skill and Tool Mapping](#4-course-unit--skill-and-tool-mapping)
5. [Complete File Structure](#5-complete-file-structure)
6. [Setup and Installation](#6-setup-and-installation)
7. [Module 1 — Data Generator](#7-module-1--data-generator)
8. [Module 2 — ML Training Pipeline](#8-module-2--ml-training-pipeline)
9. [Module 3 — Auto Follow-Up Email](#9-module-3--auto-follow-up-email)
10. [Module 4 — Employee Reminder Popup](#10-module-4--employee-reminder-popup)
11. [Module 5 — Orchestrator](#11-module-5--orchestrator)
12. [Verified Run Results](#12-verified-run-results)
13. [Review Checklist](#13-review-checklist)

---

## 1. Google Form — Exact Answers to Paste

### Project Title
```
Machine Learning-Based Missed-Lead Detection and Automated Follow-Up System for Customer Retention in Sales Pipelines
```

### Project Objective
```
To design a machine learning system that automatically identifies "missed leads" — customers who expressed interest in a product or service but did not receive a timely response from staff due to workload or oversight — and to trigger either a staff alert or an automated follow-up email, thereby reducing customer attrition caused by unattended inquiries and improving sales conversion rates.
```

### Project Domain
```
Retail and E-Commerce
```

### SDG Goal
```
SDG 8 — Decent Work and Economic Growth
```

### Type of Project
```
Application
```

---

## 2. Project Description

A **Missed-Lead Detector** is a smart assistant that watches your business inbox or CRM. When a customer contacts you expressing interest (asking for a price, demo, or availability) but your staff hasn't replied within a set time window (e.g., 24 hours), the system flags that customer as a "missed lead." It then:

1. **Sends a personalized follow-up email** to the customer in the same thread as their original message
2. **Pops up a repeating reminder** on the employee's screen, which disappears once the employee replies

---

## 3. System Architecture

```
DATA SOURCES → FEATURE ENGINEERING (Unit I)
    ↓
ML PIPELINE (Units II, III, V)
  - Supervised: LR, NB, DT, RF, XGBoost
  - Ensemble: Voting Classifier (RF + XGB + LR)
  - Unsupervised: K-Means Clustering
  - Evaluation: 5-fold CV, ROC-AUC, F1
    ↓
INFERENCE: missed_probability ≥ 0.50 → MISSED LEAD
    ↓                        ↓
auto_followup.py      employee_reminder.py
(threaded email)      (repeating popup)
```

---

## 4. Course Unit → Skill and Tool Mapping

| Unit | Topic | Usage | Tools |
|------|-------|-------|-------|
| I | Feature Engineering | `response_gap_hrs`, `intent_score`, `is_business_hours`, `gap_bucket` | Pandas, NumPy |
| II | Supervised Learning | LR, NB, DT, RF, XGBoost classification | Scikit-learn, XGBoost |
| III | Ensemble + Unsupervised | Voting Classifier + K-Means segmentation | Scikit-learn |
| IV | Neural Networks | Optional: BERT/DistilBERT intent extraction | Hugging Face |
| V | Evaluation | 5-fold Stratified CV, ROC-AUC, Feature Importance | Scikit-learn, Matplotlib |

---

## 5. Complete File Structure

```
missed_lead_detector/
├── data/
│   └── leads.csv
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── auto_followup.py
│   ├── employee_reminder.py
│   └── orchestrator.py
├── models/
│   ├── best_model.pkl
│   ├── ensemble.pkl
│   └── scaler.pkl
├── outputs/
│   ├── leads_scored.csv
│   ├── leads_segmented.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── classification_report.txt
├── logs/
│   ├── sent_leads.json
│   └── replied_leads.json
├── notebooks/
│   └── EDA.ipynb
└── MISSED_LEAD_DETECTOR_COMPLETE.md
```

---

## 6. Setup and Installation

```bash
# Install dependencies
pip install scikit-learn pandas numpy xgboost matplotlib seaborn imbalanced-learn

# Run pipeline
python src/generate_data.py       # Step 1: Generate data
python src/train_model.py         # Step 2: Train models
python src/auto_followup.py       # Step 3: Test email (demo mode)
python src/employee_reminder.py   # Step 4: Test reminder
python src/orchestrator.py        # Step 5: Full pipeline
```

---

## 7–11. Module Details

See the source files in `src/` for full implementation with inline documentation.

---

## 12. Review Checklist

### Review 1 (Phase 1)
- [ ] Project title, domain, SDG, objective
- [ ] System architecture diagram
- [ ] Unit-wise skill mapping table
- [ ] Working `generate_data.py` → `leads.csv`
- [ ] Working `train_model.py` → console output + charts
- [ ] Demo `auto_followup.py` → DEMO MODE output
- [ ] Demo `employee_reminder.py` → auto-dismiss

### Review 2 (Phase 2)
- [ ] Real CRM/WhatsApp data
- [ ] Hugging Face sentiment model (Unit IV)
- [ ] McNemar's test (Unit V)
- [ ] Dashboard (Streamlit/Flask)
- [ ] Live SMTP email

### Review 3 / Final
- [ ] Scheduled orchestrator (cron/Task Scheduler)
- [ ] EDA notebook
- [ ] Project report
- [ ] Poster/slides

---

*Document prepared by JAI VIJAI M (210425243091) and ARUNKUMAR I (210425243028), CIT Chennai, AIDS, Batch 2025-27.*
