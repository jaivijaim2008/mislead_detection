"""
email_reader.py — Missed-Lead Detector
Fetches customer inquiry emails from Gmail inbox via IMAP,
extracts features, and returns a DataFrame ready for model scoring.

-- USAGE -----------------------------------------------------------------
Requires environment variables:
    IMAP_USER   = your.email@gmail.com
    IMAP_PASS   = your-16-char-app-password
    SENDER_NAME = Your Company Name (optional, for filtering own emails)

Run standalone to see a preview of fetched emails:
    python src/email_reader.py
----------------------------------------------------------------------------
"""

import os, re, hashlib, json, socket
import imaplib
import email as email_lib
from email.header import decode_header
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd

IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
IMAP_USER = os.getenv("IMAP_USER", os.getenv("SMTP_USER", ""))
IMAP_PASS = os.getenv("IMAP_PASS", os.getenv("SMTP_PASS", ""))
SENDER_NAME = os.getenv("SENDER_NAME", "Sales Team")
MY_EMAIL = IMAP_USER.lower() if IMAP_USER else ""

# Known automated/noreply address patterns — skip these
NOREPLY_PATTERNS = [
    "noreply@", "no-reply@", "donotreply@", "do-not-reply@",
    "notifications@", "newsletter@", "mailer@", "automated@",
    "noreply-", "no-reply-", "accounts@", "support@",
]

# Known promotional / newsletter / marketing sender domains — skip these
# These send automated marketing emails, not genuine customer inquiries
PROMOTIONAL_SENDER_DOMAINS = [
    "mailin.fr", "sendinblue.com", "brevo.com", "mailchimp.com",
    "mailerlite.com", "constantcontact.com", "convertkit.com",
    "news.monetag.com", "engage.canva.com", "canva.com",
    "facebookmail.com", "priority.facebookmail.com",
    "internshala.com", "mail.internshala.com",
    "freelancer.com", "notifications.freelancer.com",
    "amazonses.com", "email.amazonses.com",
    "openrouter.ai", "customer.io",
    "codechef.com", "jobboard@codechef.com",
    "discord.com", "devpost.com",
    "dorahacks.io",
    "falconide.com",
    "blockseblock.com",
    "sparkpostmail.com",
    "notifications.google.com",
    "xt.local",
]# Known promotional subject line keywords (case-insensitive match)
PROMOTIONAL_SUBJECT_KW = [
    "sponsored", "advertisement", "promotion", "weekly digest",
    "newsletter", "you might also like", "recommended for you",
    "top picks", "don't miss", "limited time", "sale is live",
    "campaign has been sent", "your campaign",
    "monetization", "earning potential",
    "we miss you", "come back",
    "unread", "update:",
    "your profile is a good fit",
    "top internships", "internships matching",
    "build the future", "what shipped",
    "verified",
]

# OTP, verification codes, and security alerts — NOT customer inquiries
OTP_SUBJECT_PATTERNS = [
    "otp", "one time password", "one-time password",
    "verification code", "verify your", "confirm your",
    "security code", "auth code", "authentication code",
    "login code", "sign in code", "reset password",
    "password reset", "account recovery", "account verification",
    "email verification", "phone verification",
    "your code is", "code:", "use code",
    "enter the code", "enter code",
]

# Banking, financial, and transaction alerts
BANKING_SUBJECT_PATTERNS = [
    "transaction alert", "payment received", "payment confirmation",
    "order confirmation", "order placed", "order shipped",
    "delivery update", "shipping update", "track your order",
    "invoice", "receipt", "billing statement",
    "account statement", "balance alert", "low balance",
    "debit alert", "credit alert", "transfer successful",
    "upi transaction", "neft", "rtgs", "imps",
]

# Social media and app notifications
SOCIAL_NOTIFICATION_PATTERNS = [
    "new follower", "someone followed", "you have a new follower",
    "friend request", "someone tagged", "mentioned you",
    "comment on your", "replied to your", "liked your",
    "shared your", "posted in", "new message from",
    "chat message", "direct message", "someone viewed your profile",
    "connection request", "accepted your request",
    "your post was", "your comment was",
]

# Generic automated notification patterns in subject
AUTOMATED_SUBJECT_PATTERNS = [
    "automated", "system notification", "system alert",
    "do not reply", "this is an automated",
    "no reply", "noreply", "no-reply",
    "notification from", "alert from",
    "your subscription", "subscription confirmation",
    "welcome to", "account created", "account activated",
    "your account", "member since",
    "weekly report", "daily report", "monthly summary",
    "activity summary", "performance report",
]

# Known reply/sent-from addresses for our own company — skip these
OUR_DOMAINS = []  # optionally set via env var

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
SEEN_LOG = os.path.join(LOG_DIR, "seen_email_ids.json")
os.makedirs(LOG_DIR, exist_ok=True)

INTENT_WORDS = ["price", "buy", "interested", "demo", "quote", "available", "how much",
                "pricing", "purchase", "cost", "order", "subscribe", "trial"]
FILLER_WORDS = ["hello", "hi", "okay", "thanks", "thank", "just checking", "hey"]

# Marketing/promotional body keywords — if body contains these, it's likely a newsletter
PROMOTIONAL_BODY_KW = [
    "unsubscribe", "unsub", "click here to unsubscribe",
    "you are receiving this because",
    "to ensure delivery", "add to address book",
    "view in browser", "view as web page",
    "no longer want to receive",
    "manage preferences", "email preferences",
    "privacy policy", "terms of service",
    "sent to you by", "this email was sent to",
    "if you do not wish to receive",
]


# ── Helpers ──────────────────────────────────────────────────────────────

def _decode_str(val) -> str:
    """Decode email header value (handles encoded headers like =?utf-8?Q?...)."""
    if val is None:
        return ""
    decoded_parts = decode_header(val)
    result = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            try:
                result.append(part.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                result.append(part.decode("utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result)


def _extract_email_address(header_value: str) -> str:
    """Extract just the email address from 'Name <email@example.com>'."""
    match = re.search(r'<([^>]+)>', header_value)
    if match:
        return match.group(1).lower().strip()
    # If no angle brackets, treat the whole thing as an email
    return header_value.lower().strip()


def _extract_body(msg) -> str:
    """Extract plain text body from an email message."""
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body += payload.decode(charset, errors="replace")
                    except (LookupError, UnicodeDecodeError):
                        body += payload.decode("utf-8", errors="replace")

            elif content_type == "text/html" and not body:
                # Only use HTML if we haven't found plain text
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        html_text = payload.decode(charset, errors="replace")
                        # Strip HTML tags
                        body += re.sub(r'<[^>]+>', ' ', html_text)
                    except (LookupError, UnicodeDecodeError):
                        pass
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                body = payload.decode(charset, errors="replace")
            except (LookupError, UnicodeDecodeError):
                body = payload.decode("utf-8", errors="replace")

    # Clean up: remove excessive whitespace, trim forwarded/replied sections
    body = re.sub(r'\s+', ' ', body).strip()
    body = re.sub(r'^[\s>|]+', '', body, flags=re.MULTILINE)
    # Truncate to first 2000 chars to avoid signature/thread noise
    body = body[:2000]
    return body


def _is_our_own_email(sender: str, subject: str = "") -> bool:
    """Check if the email is from our own system/company.
    Allows test inquiry emails (subject starts with [TEST) to pass through.
    """
    # Allow test inquiry emails through
    if subject.startswith("[TEST"):
        return False
    if not sender:
        return True
    if MY_EMAIL and sender == MY_EMAIL:
        return True
    for domain in OUR_DOMAINS:
        if sender.endswith(domain):
            return True
    return False


def _is_promotional_email(sender_email: str, subject: str, body: str = "") -> bool:
    """
    Detect if an email is a promotional newsletter or marketing email rather than
    a genuine customer inquiry. Checks sender domain, subject, and body content.
    """
    sender_lower = sender_email.lower()
    subject_lower = subject.lower()
    body_lower = body.lower()

    # Check promotional sender domains
    for domain in PROMOTIONAL_SENDER_DOMAINS:
        if domain in sender_lower or sender_lower.endswith(domain):
            return True

    # Check promotional subject keywords
    for kw in PROMOTIONAL_SUBJECT_KW:
        if kw in subject_lower:
            return True

    # Check promotional/unsubscribe body keywords
    for kw in PROMOTIONAL_BODY_KW:
        if kw in body_lower:
            return True

    return False


def _is_automated_notification(sender_email: str, subject: str, body: str = "") -> bool:
    """
    Detect OTP, verification codes, banking alerts, social media notifications,
    and other automated emails that are NOT genuine customer inquiries.
    """
    subject_lower = subject.lower()
    body_lower = body.lower()
    sender_lower = sender_email.lower()

    # Check OTP/verification patterns in subject
    for pattern in OTP_SUBJECT_PATTERNS:
        if pattern in subject_lower:
            return True

    # Check banking/transaction patterns in subject
    for pattern in BANKING_SUBJECT_PATTERNS:
        if pattern in subject_lower:
            return True

    # Check social media notification patterns in subject
    for pattern in SOCIAL_NOTIFICATION_PATTERNS:
        if pattern in subject_lower:
            return True

    # Check generic automated notification patterns in subject
    for pattern in AUTOMATED_SUBJECT_PATTERNS:
        if pattern in subject_lower:
            return True

    # Check body for OTP patterns (common formats)
    otp_body_patterns = [
        r"your otp is\s*[:\-]?\s*\d{4,6}",
        r"your one time password is\s*[:\-]?\s*\d{4,6}",
        r"verification code[:\s]+\d{4,6}",
        r"security code[:\s]+\d{4,6}",
        r"use code[:\s]+\d{4,6}",
        r"enter[:\s]+\d{4,6}",
        r"code[:\s]+\w{4,8}",
    ]
    for pattern in otp_body_patterns:
        if re.search(pattern, body_lower):
            return True

    # Check for common automated sender patterns
    # NOTE: Do NOT include support@, help@, info@ — these are legitimate inquiry addresses
    automated_sender_patterns = [
        "otp@", "verification@", "security@", "alert@",
        "transaction@", "payment@", "order@", "shipping@",
        "notification@", "notify@", "updates@",
        "accounts@", "billing@", "invoice@",
        "hello@", "hi@",  # Generic automated greetings
    ]
    for pattern in automated_sender_patterns:
        if pattern in sender_lower:
            return True

    # Check for numeric sender addresses (common for OTP systems)
    if re.match(r'^\d+@', sender_lower):
        return True

    return False


def _is_auto_reply(msg) -> bool:
    """Detect auto-replies and bounces."""
    auto_headers = [
        "Auto-Submitted", "X-Autoreply", "X-Auto-Response-Suppress",
        "Precedence", "X-Precedence"
    ]
    for h in auto_headers:
        val = str(msg.get(h, "")).lower()
        if val in ("auto-replied", "auto-generated", "bulk", "junk", "list"):
            return True
    subject = str(msg.get("Subject", "")).lower()
    if any(kw in subject for kw in ("out of office", "auto-reply", "autoreply",
                                      "auto reply", "returned mail", "undeliverable",
                                      "mail delivery failed")):
        return True
    return False


def _compute_lead_id(sender: str, subject: str, date_str: str) -> str:
    """Create a deterministic unique lead ID from email metadata."""
    raw = f"{sender}|{subject}|{date_str}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:8].upper()
    return f"E-{digest}"


def _load_seen_ids() -> set:
    if os.path.exists(SEEN_LOG):
        with open(SEEN_LOG) as f:
            return set(json.load(f))
    return set()


def _save_seen_ids(ids: set):
    with open(SEEN_LOG, "w") as f:
        json.dump(sorted(ids), f, indent=2)


# ── Main Fetch ──────────────────────────────────────────────────────────

def fetch_customer_emails(max_emails: int = 50,
                          search_since_days: int = 30,
                          min_body_length: int = 5) -> pd.DataFrame:
    """
    Connect to Gmail IMAP, fetch recent customer inquiry emails,
    and return a DataFrame with the features the model expects.

    Args:
        max_emails: Max customer emails to fetch
        search_since_days: Look back this many days
        min_body_length: Minimum body length to consider (filter noise)

    Returns:
        pd.DataFrame with columns matching the model's expected input features
    """
    if not IMAP_USER or not IMAP_PASS:
        print("[email_reader] ERROR: IMAP_USER and IMAP_PASS env vars not set.")
        print("[email_reader] Run: export IMAP_USER='your.email@gmail.com'")
        print("[email_reader]     export IMAP_PASS='your-16-char-app-password'")
        return pd.DataFrame()

    print(f"[email_reader] Connecting to {IMAP_HOST}:{IMAP_PORT} as {IMAP_USER} ...")

    # Guard against indefinitely-hanging IMAP connections (network congestion,
    # firewall drop, etc.).  30 s is generous for a TLS handshake + login.
    socket.setdefaulttimeout(30)

    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        mail.login(IMAP_USER, IMAP_PASS)
        print("[email_reader] Connected successfully.")
    except imaplib.IMAP4.error as e:
        print(f"[email_reader] LOGIN FAILED: {e}")
        print("[email_reader] Check your credentials and ensure App Password is correct.")
        socket.setdefaulttimeout(None)
        return pd.DataFrame()
    except Exception as e:
        print(f"[email_reader] Connection error: {e}")
        socket.setdefaulttimeout(None)
        return pd.DataFrame()

    # Select inbox
    status, _ = mail.select("INBOX")
    if status != "OK":
        print(f"[email_reader] Could not select INBOX (status={status})")
        mail.logout()
        return pd.DataFrame()

    # Search for messages from the last N days
    if search_since_days > 0:
        since_date = (datetime.now(timezone.utc) - timedelta(days=search_since_days)
                     ).strftime("%d-%b-%Y")
        result, data = mail.search(None, f"(SINCE {since_date})")
    else:
        result, data = mail.search(None, "ALL")

    if result != "OK":
        print(f"[email_reader] Search failed: {result}")
        mail.logout()
        return pd.DataFrame()

    email_ids = data[0].split() if data[0] else []
    total_found = len(email_ids)
    print(f"[email_reader] Found {total_found} emails in inbox.")

    # Limit how many we process
    email_ids = email_ids[-max_emails * 3:]  # fetch more to account for filtering
    seen_ids = _load_seen_ids()
    new_seen = set(seen_ids)
    rows = []

    for eid in reversed(email_ids):
        if len(rows) >= max_emails:
            break

        try:
            # ── Stage 1: fetch headers only (cheap) ──────────────────────────
            # Download just From, Subject, Message-ID, and Date to decide
            # whether this email is worth fetching the full body for.
            h_status, h_data = mail.fetch(
                eid,
                "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT MESSAGE-ID DATE)])"
            )
            if h_status != "OK" or not h_data or not h_data[0]:
                continue

            hdr_block = h_data[0][1] if isinstance(h_data[0], tuple) else b""
            hdr_msg = email_lib.message_from_bytes(hdr_block)

            subject     = _decode_str(hdr_msg.get("Subject", ""))
            from_hdr    = _decode_str(hdr_msg.get("From", ""))
            sender_email = _extract_email_address(from_hdr)
            date_hdr    = hdr_msg.get("Date", "")
            msg_id      = str(hdr_msg.get("Message-ID", "")).strip()

            # ── Header-level filters (no body download needed) ───────────────
            if not sender_email or _is_our_own_email(sender_email, subject):
                continue
            if msg_id and msg_id in seen_ids:
                continue
            is_automated = any(pattern in sender_email for pattern in NOREPLY_PATTERNS)
            if is_automated:
                continue
            # Quick subject-only promotional check before body download
            if _is_promotional_email(sender_email, subject):
                continue
            if _is_automated_notification(sender_email, subject):
                continue

            # ── Stage 2: full fetch (only if headers pass) ───────────────────
            status, msg_data = mail.fetch(eid, "(RFC822)")
            if status != "OK":
                continue

            raw_email = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw_email)

            # Re-read headers from the full message (more reliable)
            subject      = _decode_str(msg.get("Subject", ""))
            from_hdr     = _decode_str(msg.get("From", ""))
            sender_email = _extract_email_address(from_hdr)
            date_hdr     = msg.get("Date", "")
            msg_id       = str(msg.get("Message-ID", "")).strip()

            # Auto-reply check needs full message headers
            if _is_auto_reply(msg):
                continue

            body = _extract_body(msg)
            if len(body) < min_body_length:
                continue

            # Full body-aware promotional / automated checks
            if _is_promotional_email(sender_email, subject, body):
                continue
            if _is_automated_notification(sender_email, subject, body):
                continue

            # ── Parse timestamp ──
            try:
                dt = parsedate_to_datetime(date_hdr)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                msg_hour = dt.hour
                received_ts = dt.timestamp()
            except Exception:
                msg_hour = 12
                received_ts = datetime.now(timezone.utc).timestamp()

            # ── Compute features ──
            now_ts = datetime.now(timezone.utc).timestamp()
            response_gap = max(0, (now_ts - received_ts) / 3600)  # hours

            # Intent score
            body_lower = body.lower()
            intent_score = sum(body_lower.count(w) for w in INTENT_WORDS)
            is_high_intent = intent_score >= 1 or any(
                w in body_lower for w in ["price", "buy", "interested", "demo", "quote",
                                           "available", "order", "purchase", "cost"]
            )

            # Message length
            msg_len = len(body)

            # Previous contacts: check if we've seen this sender before
            prev = 0

            # Parse customer name from From header
            name_match = re.match(r'^([^<]+)', from_hdr)
            customer_name = name_match.group(1).strip() if name_match else sender_email.split('@')[0]

            lead_id = _compute_lead_id(sender_email, subject, date_hdr)

            # Remember the message ID so we don't re-process
            if msg_id:
                new_seen.add(msg_id)

            rows.append({
                "lead_id": lead_id,
                "channel": "Email",
                "message_text": body[:300],  # truncate for storage
                "message_hour": msg_hour,
                "message_length": msg_len,
                "high_intent_flag": int(is_high_intent),
                "prev_contacts": prev,
                "response_gap_hrs": round(response_gap, 2),
                # Extra metadata for sending follow-up
                "_customer_email": sender_email,
                "_customer_name": customer_name,
                "_subject": subject,
                "_message_id": msg_id,
                "_received_time": datetime.fromtimestamp(received_ts, tz=timezone.utc)
                                  .strftime("%Y-%m-%d %H:%M UTC"),
            })

        except Exception as e:
            print(f"[email_reader] Error processing email {eid}: {e}")
            continue

    mail.logout()
    # Restore default socket timeout so rest of application isn't affected
    socket.setdefaulttimeout(None)

    # Save seen IDs
    _save_seen_ids(new_seen)

    df = pd.DataFrame(rows)
    print(f"\n[email_reader] Fetched {len(rows)} customer inquiry emails "
          f"(filtered from {total_found} total inbox emails)")

    if len(df) > 0:
        print(f"[email_reader] Average response gap: {df['response_gap_hrs'].mean():.1f}h")
        print(f"[email_reader] High-intent count: {df['high_intent_flag'].sum()}/{len(df)}")
        print(f"[email_reader] Senders: {', '.join(df['_customer_email'].unique()[:5])}...")
        print(f"[email_reader] To send real follow-ups, ensure SMTP env vars are set.")

    return df


def preview_emails(n: int = 10):
    """Print a readable preview of the latest customer emails."""
    df = fetch_customer_emails(max_emails=n, search_since_days=30, min_body_length=3)
    if df.empty:
        print("No customer emails found in inbox.")
        return

    print(f"\n{'='*70}")
    print(f"  LATEST {len(df)} CUSTOMER INQUIRIES — PREVIEW")
    print(f"{'='*70}")
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"\n--- Email #{i+1}: {row['lead_id']} ---")
        print(f"  From      : {row['_customer_name']} <{row['_customer_email']}>")
        print(f"  Subject   : {row['_subject']}")
        print(f"  Received  : {row['_received_time']}")
        print(f"  Gap       : {row['response_gap_hrs']:.1f} hours")
        print(f"  Intent    : {'HIGH' if row['high_intent_flag'] else 'LOW'} "
              f"(score: {sum(row['message_text'].lower().count(w) for w in INTENT_WORDS)})")
        body_preview = row['message_text'][:120].replace('\n', ' ')
        print(f"  Body      : \"{body_preview}...\"")


if __name__ == "__main__":
    preview_emails(15)
