# Lemma AI Prompt — Missed-Lead Detector Full Completion

> Paste this entire prompt into Lemma AI to make the project production-complete.

---

## System Context

You are working on a **Missed-Lead Detector** — an AI-powered sales pipeline retention system built with Python 3.11. The project lives at `missed_lead_detector/` and uses Streamlit for the dashboard, scikit-learn/XGBoost/PyTorch for ML, IMAP/SMTP for Gmail integration, and GitHub Actions for scheduling.

**Current state:** The core ML pipeline, email integration, smart reply engine, Streamlit dashboard, and deployment infrastructure are built and working. Your job is to **complete all missing pieces** to make this a production-ready, fully-featured application.

---

## Project File Structure (Current)

```
missed_lead_detector/
├── src/
│   ├── config.py              ✅ Business config (env var overrides)
│   ├── dashboard.py           ✅ Streamlit dashboard (6 pages, Nebula glassmorphism UI)
│   ├── orchestrator.py        ✅ Pipeline runner (demo/live/preview)
│   ├── train_model.py         ✅ ML + DL training (8 models)
│   ├── deep_learning.py       ✅ PyTorch ResNet-style classifier
│   ├── xgb_tuning.py          ✅ Optuna hyperparameter tuning
│   ├── email_reader.py        ✅ IMAP Gmail reader with filtering
│   ├── auto_followup.py       ✅ SMTP threaded follow-up with retry
│   ├── smart_reply_engine.py  ✅ Intent-aware reply templates
│   ├── hf_intent_model.py     ✅ HuggingFace zero-shot intent + sentiment
│   ├── inbox_monitor.py       ✅ Continuous inbox monitoring
│   ├── notifications.py       ✅ Multi-channel notifications
│   ├── daily_digest.py        ✅ Daily email digest
│   ├── employee_reminder.py   ✅ Repeating popup reminders
│   ├── generate_data.py       ✅ Synthetic data generator
│   ├── merge_kaggle_data.py   ✅ Kaggle dataset merger
│   ├── plot_model_comparison.py ✅ AUC comparison chart
│   └── mcnemar_test.py        ✅ McNemar's statistical test
├── data/                      ✅ leads.csv, leads_merged.csv, kaggle/
├── models/                    ✅ 7 trained artifacts (.pkl, .pt)
├── outputs/                   ✅ Charts, CSVs, reports, EDA/
├── logs/                      ✅ JSON logs (sent, replies, notifications)
├── notebooks/                 ✅ EDA.py (script, not notebook)
├── .github/workflows/         ✅ inbox-monitor.yml, daily-digest.yml
├── Dockerfile                 ✅ Docker deployment
├── render.yaml                ✅ Render.com config
├── requirements.txt           ✅ Full deps
├── requirements-ci.txt        ✅ CI deps
└── README.md                  ✅ Documentation
```

---

## TASK 1: Unit Tests & CI Pipeline

Create comprehensive unit tests for ALL source modules:

**Files to create:**
- `tests/__init__.py`
- `tests/conftest.py` — shared fixtures (sample DataFrames, mock SMTP, mock IMAP, sample lead dicts)
- `tests/test_config.py` — test config loading, env var overrides, defaults
- `tests/test_generate_data.py` — test data generation produces correct schema, distributions
- `tests/test_train_model.py` — test feature engineering, model training, AUC > 0.85
- `tests/test_deep_learning.py` — test model architecture, forward pass, save/load
- `tests/test_smart_reply_engine.py` — test intent detection for all 8 intents, template generation, placeholder filling
- `tests/test_email_reader.py` — test email filtering (promotional, OTP, auto-reply detection), feature extraction
- `tests/test_auto_followup.py` — test email building, dedup logic, retry mechanism
- `tests/test_inbox_monitor.py` — test scoring, scan summary, log persistence
- `tests/test_notifications.py` — test dashboard notification store, mark read
- `tests/test_mcnemar_test.py` — test McNemar's statistic calculation, contingency table
- `tests/test_hf_intent_model.py` — test keyword fallback classification, sentiment lexicon
- `tests/test_orchestrator.py` — test demo pipeline scoring, artifact loading
- `tests/test_daily_digest.py` — test digest building, HTML generation

**Requirements:**
- Use `pytest` as the test runner
- Add `pytest` and `pytest-cov` to `requirements-ci.txt`
- Mock external services (SMTP, IMAP, tkinter) — tests must pass without real credentials
- Achieve >80% code coverage
- Create `tests/test_runner.py` or use pytest.ini for configuration
- Create a GitHub Actions workflow `.github/workflows/test.yml` that runs tests on every push/PR

**Example conftest.py fixture:**
```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_leads_df():
    return pd.DataFrame({
        "lead_id": ["L00001", "L00002", "L00003"],
        "channel": ["Email", "WhatsApp", "Phone Inquiry"],
        "message_text": ["price of course", "interested in demo", "hello"],
        "message_hour": [10, 14, 21],
        "message_length": [18, 24, 5],
        "high_intent_flag": [1, 1, 0],
        "prev_contacts": [0, 2, 1],
        "response_gap_hrs": [48.5, 12.3, 120.0],
        "replied": [0, 1, 0],
    })

@pytest.fixture
def sample_lead_dict():
    return {
        "lead_id": "L00042",
        "customer_email": "priya@example.com",
        "customer_name": "Priya",
        "channel": "Email",
        "subject": "Course Pricing",
        "original_message_id": "<abc123@mail.example.com>",
    }
```

---

## TASK 2: System Architecture Diagram

Create `tools/generate_architecture.py` that generates a professional `outputs/system_architecture.png` diagram showing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MISSED-LEAD DETECTOR                         │
│                   System Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA SOURCES          ML PIPELINE           ACTIONS            │
│  ────────────          ───────────           ───────            │
│  ┌──────────┐     ┌──────────────────┐   ┌──────────────┐      │
│  │ Gmail    │────▶│ Feature Engine   │──▶│ Auto-Reply   │      │
│  │ (IMAP)   │     │ (9 features)     │   │ (SMTP)       │      │
│  └──────────┘     └──────────────────┘   └──────────────┘      │
│       │                  │                      │               │
│       │           ┌──────┴──────┐        ┌──────┴──────┐       │
│       │           │ ML Ensemble │        │ Desktop     │       │
│       │           │ (RF+XGB+LR) │        │ Reminders   │       │
│       │           └──────┬──────┘        └─────────────┘       │
│       │                  │                      │               │
│       │           ┌──────┴──────┐        ┌──────┴──────┐       │
│       │           │ DL Model    │        │ Dashboard   │       │
│       │           │ (PyTorch)   │        │ (Streamlit) │       │
│       │           └──────┬──────┘        └─────────────┘       │
│       │                  │                      │               │
│       │           ┌──────┴──────┐        ┌──────┴──────┐       │
│       │           │ Grand       │        │ Email       │       │
│       │           │ Ensemble    │        │ Digest      │       │
│       │           └─────────────┘        └─────────────┘       │
│       │                                                         │
│  ┌──────────┐     ┌──────────────────┐                         │
│  │ Kaggle   │────▶│ Smart Reply      │   MONITORING            │
│  │ Datasets │     │ Engine           │   ──────────            │
│  └──────────┘     │ (Intent + NLP)   │   ┌──────────────┐      │
│                   └──────────────────┘   │ GitHub Actions│      │
│  SCHEDULING                              │ (hourly scan) │      │
│  ──────────                              └──────────────┘      │
│  ┌──────────┐                              ┌──────────────┐     │
│  │ Cron/    │                              │ Notifications│     │
│  │ Actions  │─────────────────────────────▶│ (3-channel)  │     │
│  └──────────┘                              └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Use `matplotlib` with professional styling (dark theme matching the dashboard), include module names as labels, and add arrows showing data flow. The diagram should be publication-quality.

---

## TASK 3: REST API for External Integration

Create `src/api.py` — a FastAPI-based REST API that exposes the system's capabilities to external tools:

**Endpoints to implement:**

```
POST /api/v1/score          — Score a single lead or batch of leads
POST /api/v1/reply/preview  — Preview smart reply for a lead
POST /api/v1/reply/send     — Send follow-up email
GET  /api/v1/leads           — List scored leads with filtering
GET  /api/v1/leads/{id}      — Get lead details
GET  /api/v1/stats           — Pipeline statistics
POST /api/v1/scan            — Trigger an inbox scan
GET  /api/v1/health          — Health check
```

**Requirements:**
- Use FastAPI with Pydantic models for request/response validation
- Add API key authentication via `X-API-Key` header (key stored in env var `API_KEY`)
- Rate limiting: 100 requests per minute per API key
- OpenAPI docs auto-generated at `/docs`
- Add `fastapi` and `uvicorn` to `requirements.txt`
- Create `src/api_models.py` for Pydantic schemas
- Add CORS middleware (configurable origins)
- Log all API requests to `logs/api_access.log`

**Example Pydantic model:**
```python
from pydantic import BaseModel, EmailStr
from typing import Optional

class LeadScoreRequest(BaseModel):
    channel: str
    message_text: str
    message_hour: int = 12
    message_length: Optional[int] = None
    high_intent_flag: Optional[int] = None
    prev_contacts: int = 0
    response_gap_hrs: float = 24.0

class LeadScoreResponse(BaseModel):
    lead_id: str
    missed_probability: float
    predicted_missed: bool
    high_intent: bool
    recommended_action: str

class ReplyPreviewRequest(BaseModel):
    customer_name: str
    customer_email: EmailStr
    subject: str
    message_text: str
    channel: str = "Email"

class ReplyPreviewResponse(BaseModel):
    reply_subject: str
    reply_body: str
    detected_intent: str
    intent_scores: dict
```

---

## TASK 4: Database Integration (SQLite)

Replace JSON file storage with SQLite for production reliability:

**Create `src/database.py`:**
```python
"""
database.py — SQLite storage backend for leads, replies, notifications.
Replaces JSON file reads/writes with proper database operations.
"""
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "missed_leads.db")

class Database:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS leads (
                    lead_id TEXT PRIMARY KEY,
                    channel TEXT,
                    message_text TEXT,
                    message_hour INTEGER,
                    message_length INTEGER,
                    high_intent_flag INTEGER,
                    prev_contacts INTEGER,
                    response_gap_hrs REAL,
                    missed_probability REAL,
                    predicted_missed INTEGER,
                    customer_email TEXT,
                    customer_name TEXT,
                    subject TEXT,
                    received_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS replies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lead_id TEXT,
                    customer_email TEXT,
                    customer_name TEXT,
                    reply_subject TEXT,
                    reply_body TEXT,
                    detected_intent TEXT,
                    sent_at TEXT,
                    status TEXT DEFAULT 'sent',
                    FOREIGN KEY (lead_id) REFERENCES leads(lead_id)
                );

                CREATE TABLE IF NOT EXISTS followup_status (
                    lead_id TEXT PRIMARY KEY,
                    customer_name TEXT,
                    customer_email TEXT,
                    auto_replied INTEGER DEFAULT 0,
                    auto_replied_at TEXT,
                    human_followed_up INTEGER DEFAULT 0,
                    human_followed_up_at TEXT,
                    overdue_notified INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT,
                    title TEXT,
                    message TEXT,
                    customer_name TEXT,
                    lead_id TEXT,
                    read INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS scan_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scanned INTEGER,
                    missed_detected INTEGER,
                    replied INTEGER,
                    skipped INTEGER,
                    timestamp TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_leads_predicted ON leads(predicted_missed);
                CREATE INDEX IF NOT EXISTS idx_replies_lead ON replies(lead_id);
                CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(read);
            """)
```

**Then update these files to use `database.py` instead of JSON:**
- `inbox_monitor.py` — save scored leads, reply logs, followup status to DB
- `auto_followup.py` — check sent IDs from DB instead of `sent_leads.json`
- `notifications.py` — store notifications in DB instead of `notifications.json`
- `dashboard.py` — read from DB instead of CSV/JSON files
- `daily_digest.py` — query DB for last 24h stats

**Important:** Keep backward compatibility — if the DB doesn't exist, fall back to JSON files. Add a migration script `tools/migrate_json_to_db.py` that imports existing JSON data into the database.

---

## TASK 5: Enhanced Dashboard Features

Add these features to `src/dashboard.py`:

### 5a. Real-Time Auto-Refresh
Add `st_autorefresh` (from `streamlit-autorefresh`) to poll for new data every 30 seconds when on the Command Center page.

### 5b. K-Means Segment Visualization
Add a new section to the "Lead Explorer" page showing the K-Means customer segments:
- Segment names: "High-Intent-Missed", "Low-Intent", "Already-Converted"
- Pie chart showing segment distribution
- Table showing segment characteristics (avg gap, avg intent, count)

### 5c. Model Performance Deep Dive
Add a new page "Model Deep Dive" with:
- Interactive confusion matrix (Plotly heatmap)
- Per-class precision/recall/F1 bar chart
- ROC curve comparison for all 8 models
- Feature importance horizontal bar chart
- Learning curves (train vs validation loss over epochs)

### 5d. A/B Reply Template Testing
Add a page "Reply A/B Tests" where users can:
- Select two template variants to compare
- See mock replies side-by-side for sample emails
- Track which variant performs better (open rate proxy: reply_received count)

### 5e. Export & Reports
Add "Export" buttons on each page:
- Export leads as CSV/Excel
- Export model comparison as PDF report
- Export daily digest as HTML file

### 5f. Settings Page Enhancement
- Add course management (add/edit/remove courses dynamically)
- Add batch schedule editor
- Add template preview with live editing
- Show environment variable status with connection test buttons

---

## TASK 6: Sentiment Integration into Pipeline

Integrate the existing `hf_intent_model.py` sentiment analysis into the main pipeline:

### 6a. Modify `inbox_monitor.py`:
```python
# After scoring each email, also run sentiment analysis
from hf_intent_model import analyze_sentiment

sentiment = analyze_sentiment(row["message_text"])
df.at[idx, "sentiment"] = sentiment["sentiment"]
df.at[idx, "sentiment_score"] = sentiment["score"]
```

### 6b. Modify `smart_reply_engine.py`:
- Use sentiment to adjust reply tone:
  - **Negative sentiment** → more empathetic, apologetic tone, escalation keywords
  - **Positive sentiment** → warm, encouraging, cross-sell opportunity
  - **Neutral sentiment** → standard professional tone

### 6c. Add to Dashboard:
- Sentiment distribution chart (pie chart)
- Sentiment trend over time (line chart)
- Sentiment by channel breakdown
- Alert when negative sentiment exceeds 30%

### 6d. Modify `daily_digest.py`:
- Include sentiment summary in the digest email
- Highlight any negative-sentiment leads that need immediate attention

---

## TASK 7: Rate Limiting & Safety Guards

Add safety mechanisms to prevent abuse:

### 7a. Rate Limiter (`src/rate_limiter.py`):
```python
"""
Rate limiter to prevent auto-reply spam.
Configurable per-email and per-day limits.
"""
import time
from collections import defaultdict
from config import MAX_REPLIES_PER_HOUR, MAX_REPLIES_PER_DAY

class RateLimiter:
    def __init__(self, max_per_hour=20, max_per_day=200):
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self._hourly: dict[str, list[float]] = defaultdict(list)
        self._daily: dict[str, list[float]] = defaultdict(list)

    def can_send(self, recipient: str) -> tuple[bool, str]:
        now = time.time()
        # Clean old entries
        self._hourly[recipient] = [t for t in self._hourly[recipient] if now - t < 3600]
        self._daily[recipient] = [t for t in self._daily[recipient] if now - t < 86400]

        if len(self._hourly[recipient]) >= self.max_per_hour:
            return False, f"Rate limit: {self.max_per_hour} emails/hour to {recipient}"
        if len(self._daily[recipient]) >= self.max_per_day:
            return False, f"Rate limit: {self.max_per_day} emails/day to {recipient}"
        return True, "OK"

    def record_send(self, recipient: str):
        now = time.time()
        self._hourly[recipient].append(now)
        self._daily[recipient].append(now)
```

### 7b. Integrate into `auto_followup.py`:
- Check rate limit before sending
- Log rate-limited attempts to `logs/rate_limited.json`

### 7c. Config additions in `config.py`:
```python
MAX_REPLIES_PER_HOUR = overrides.get("MAX_REPLIES_PER_HOUR", 20)
MAX_REPLIES_PER_DAY = overrides.get("MAX_REPLIES_PER_DAY", 200)
COOLDOWN_AFTER_REPLY_MINS = overrides.get("COOLDOWN_AFTER_REPLY_MINS", 5)
```

### 7d. Add to Dashboard Settings page:
- Display current rate limits
- Allow adjusting limits via the settings form

---

## TASK 8: Model Auto-Retraining Trigger

Create `src/auto_retrain.py` that triggers model retraining when data drift is detected:

### 8a. Data Drift Detection:
```python
"""
Monitor for data drift in incoming emails.
Triggers retraining when feature distributions shift significantly.
"""
import numpy as np
from scipy import stats

def detect_drift(reference_stats: dict, current_stats: dict, threshold=0.05) -> dict:
    """Compare reference and current feature distributions using KS test."""
    drift_report = {}
    for feature in reference_stats:
        if feature in current_stats:
            ks_stat, p_value = stats.ks_2samp(
                reference_stats[feature], current_stats[feature]
            )
            drift_report[feature] = {
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "drifted": p_value < threshold,
            }
    return drift_report
```

### 8b. Auto-Retrain Logic:
- After each inbox scan, compare last 100 scored emails against training data statistics
- If >30% of features show drift, log a warning
- If >50% show drift, automatically run `train_model.py`
- Send notification to dashboard: "Model retrained due to data drift"

### 8c. Version Tracking:
- Save model versions with timestamps: `models/v_20240101_ensemble.pkl`
- Keep last 5 model versions
- Dashboard shows model version history and performance per version

---

## TASK 9: EDA Jupyter Notebook

Convert `notebooks/EDA.py` to a proper Jupyter notebook `notebooks/EDA.ipynb`:

**Sections to include:**
1. **Setup & Data Loading** — imports, load leads.csv, display first 5 rows
2. **Data Overview** — shape, dtypes, missing values, describe()
3. **Target Analysis** — replied distribution with pie chart
4. **Channel Analysis** — distribution, missed rate by channel
5. **Response Gap Analysis** — histogram, box plot, correlation with target
6. **Intent Analysis** — keyword frequency, intent score distribution
7. **Time Analysis** — hourly distribution, business hours vs after hours
8. **Feature Correlation** — heatmap with annotations
9. **Feature Importance** — from trained Random Forest model
10. **Statistical Tests** — chi-square for categorical features, t-tests for numerical
11. **Key Insights** — markdown cells summarizing findings
12. **Recommendations** — what to improve in the model

Each section should have markdown explanations, code cells, and inline visualizations. The notebook should be runnable end-to-end.

---

## TASK 10: Comprehensive Documentation

### 10a. Update `README.md`:
- Add badges: tests passing, coverage, Python version, Streamlit
- Add "Architecture Overview" section with the generated diagram
- Add "API Reference" section (after Task 3)
- Add "Deployment Guide" for Render, Docker, and Streamlit Cloud
- Add "Troubleshooting" section for common issues
- Add "Contributing" guidelines
- Add "Performance Benchmarks" table

### 10b. Create `docs/API.md`:
- Full API documentation with request/response examples
- Authentication guide
- Rate limiting documentation
- Error codes reference

### 10c. Create `docs/DEPLOYMENT.md`:
- Step-by-step deployment for:
  - Streamlit Cloud (current method)
  - Render.com (current Docker method)
  - Docker Compose (local development)
  - AWS EC2 / GCP Cloud Run
- Environment variable reference table
- Gmail App Password setup guide

### 10d. Create `docs/MODULES.md`:
- Detailed explanation of each module's purpose
- Data flow diagram
- Configuration options for each module
- Example usage for each module

---

## TASK 11: Performance Optimization

### 11a. Dashboard Caching:
- Add `@st.cache_resource` for model loading (load once, not on every rerun)
- Add `@st.cache_data` with appropriate TTL for all data loading
- Use `st.fragment` for independent dashboard sections to enable partial reruns

### 11b. Email Processing:
- Batch IMAP fetch operations (fetch all UIDs first, then batch-fetch messages)
- Use connection pooling for SMTP sends
- Add connection timeout handling

### 11c. Model Inference:
- Pre-compute feature encodings where possible
- Cache LabelEncoder fit on training data and reuse at inference
- Add model warm-up on startup

---

## TASK 12: Security Hardening

### 12a. Secrets Management:
- Never log SMTP/IMAP passwords
- Mask sensitive env vars in dashboard (show `***` for last 4 chars only)
- Add `.env.example` with placeholder values

### 12b. Input Validation:
- Validate all email addresses before sending
- Sanitize message text (strip HTML injection attempts)
- Validate lead_id format (must match `L\d{5}` or `E-[A-F0-9]{8}`)

### 12c. HTTPS Enforcement:
- Add `--server.enableCORS=false` and `--server.enableXsrfProtection=false` only for development
- Add nginx reverse proxy config for production SSL termination

---

## Implementation Order

Please complete tasks in this order:
1. **TASK 1** (Tests) — Foundation for all other changes
2. **TASK 4** (Database) — Core infrastructure upgrade
3. **TASK 7** (Rate Limiting) — Safety before adding features
4. **TASK 3** (REST API) — External integration layer
5. **TASK 2** (Architecture Diagram) — Documentation
6. **TASK 6** (Sentiment Integration) — ML enhancement
7. **TASK 8** (Auto-Retraining) — ML operations
8. **TASK 5** (Dashboard Enhancements) — UI improvements
9. **TASK 9** (EDA Notebook) — Analysis documentation
10. **TASK 10** (Documentation) — Final documentation
11. **TASK 11** (Performance) — Optimization
12. **TASK 12** (Security) — Hardening

---

## Quality Requirements

- **Every new Python file** must have a module docstring explaining its purpose
- **Every function** must have a docstring with Args/Returns
- **No hardcoded paths** — use `os.path.join(BASE, ...)` pattern from existing code
- **No hardcoded secrets** — always read from env vars or `config.py`
- **Follow existing code style** — match the import order, comment style, and formatting of existing files
- **All tests must pass** before considering a task complete
- **Backward compatible** — existing functionality must not break
- **The dashboard must remain deployable** to Streamlit Cloud without changes to the deployment process

## Tech Stack Constraints

- Python 3.11
- Streamlit (dashboard, no React/Next.js)
- FastAPI (REST API only)
- SQLite (database, no PostgreSQL for simplicity)
- PyTorch (deep learning, CPU-only for cloud deployment)
- scikit-learn, XGBoost, Optuna (ML)
- Plotly + Matplotlib (visualization)
- No paid APIs or services (everything must work with free tier)
