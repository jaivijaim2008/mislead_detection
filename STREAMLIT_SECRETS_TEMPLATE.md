# Streamlit Cloud Secrets Setup

Copy the content below into **Streamlit Cloud → Settings → Secrets**.
Replace **all `your-*` values** with your real credentials.

## How to get Gmail App Password

1. Go to **Google Account → Security → 2-Step Verification** (enable it)
2. Go to **Security → App Passwords** (search for it if not visible)
3. Generate a password for **"Mail"** on **"Windows Computer"**
4. Copy the 16-character password — use it for both `IMAP_PASS` and `SMTP_PASS`

## Secrets to Add

```toml
# ── Dashboard Login ────────────────────────────────────
AUTH_USER = "admin"
AUTH_PASS = "your-strong-password-here"

# ── Gmail IMAP (read inbox) ────────────────────────────
IMAP_USER = "your-email@gmail.com"
IMAP_PASS = "your-16-char-app-password"

# ── Gmail SMTP (send auto-replies) ─────────────────────
SMTP_USER = "your-email@gmail.com"
SMTP_PASS = "your-16-char-app-password"

# ── Notification Recipient ─────────────────────────────
NOTIFY_EMAIL = "your-email@gmail.com"
SENDER_NAME = "Sales Team"
```

## Environment Variables (Alternative)

If you prefer, you can also set these as environment variables in Streamlit Cloud:

| Variable | Description |
|----------|-------------|
| `AUTH_USER` | Dashboard username |
| `AUTH_PASS` | Dashboard password |
| `IMAP_USER` | Gmail address for reading inbox |
| `IMAP_PASS` | Gmail App Password for IMAP |
| `SMTP_USER` | Gmail address for sending replies |
| `SMTP_PASS` | Gmail App Password for SMTP |
| `NOTIFY_EMAIL` | Where to send alert emails |
| `SENDER_NAME` | Sender name shown in auto-replies |
