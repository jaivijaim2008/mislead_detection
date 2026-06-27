"""List all emails in Gmail INBOX with sender and subject."""
import imaplib
import email as email_lib
from email.header import decode_header
import os

IMAP_USER = os.getenv("IMAP_USER", "missedlead.detector@gmail.com")
IMAP_PASS = os.getenv("IMAP_PASS", "qxyg ppyp hmke resq")

def decode_str(val):
    if val is None:
        return ""
    parts = decode_header(val)
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result)

mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
mail.login(IMAP_USER, IMAP_PASS)
mail.select("INBOX")
status, data = mail.search(None, "ALL")
ids = data[0].split()

print(f"Total emails in INBOX: {len(ids)}\n")

for i, eid in enumerate(ids):
    status, msg_data = mail.fetch(eid, "(RFC822)")
    if status != "OK":
        continue
    msg = email_lib.message_from_bytes(msg_data[0][1])
    from_hdr = decode_str(msg.get("From", ""))
    subject = decode_str(msg.get("Subject", ""))
    date = msg.get("Date", "")[:25]
    print(f"[{i+1:2d}] From: {from_hdr}")
    print(f"     Subject: {subject}")
    print(f"     Date: {date}")
    print()

mail.logout()
