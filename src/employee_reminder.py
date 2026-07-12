"""
employee_reminder.py — Missed-Lead Detector
Repeating popup reminder for the employee managing client emails.
- Pops up every INTERVAL_SECONDS until the employee marks the lead replied.
- Auto-dismisses instantly once mark_replied() is called for that lead.
- Headless-safe: falls back to terminal banner when tkinter is unavailable.
"""

import threading, time, sys, os, json

INTERVAL_SECONDS = 60
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "replied_leads.json")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

try:
    import tkinter as tk
    from tkinter import messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Disable GUI when explicitly running headless (e.g. Windows Task Scheduler).
# run_daily.bat sets STREAMLIT_SERVER_HEADLESS=true; DISPLAY absence covers Linux.
if os.environ.get("STREAMLIT_SERVER_HEADLESS", "").lower() in ("1", "true"):
    GUI_AVAILABLE = False
if not os.environ.get("DISPLAY", "") and sys.platform != "win32":
    GUI_AVAILABLE = False

def _load_replied() -> set:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return set(json.load(f))
    return set()

def _save_replied(replied: set):
    with open(LOG_FILE, "w") as f:
        json.dump(sorted(replied), f, indent=2)

def mark_replied(lead_id: str):
    """Call this when the employee actually replies to the customer."""
    replied = _load_replied()
    replied.add(lead_id)
    _save_replied(replied)
    print(f"[reminder] Lead {lead_id} marked as replied - popup dismissed.")

def is_replied(lead_id: str) -> bool:
    return lead_id in _load_replied()

def _print_banner(lead: dict):
    print(
        f"\n{'!'*60}\n"
        f"  MISSED LEAD REMINDER\n"
        f"  Lead ID   : {lead['lead_id']}\n"
        f"  Customer  : {lead.get('customer_name','N/A')}\n"
        f"  Channel   : {lead.get('channel','N/A')}\n"
        f"  Overdue   : {lead.get('gap_hrs', '?')} hrs without reply\n"
        f"  ACTION    : Please respond NOW or call mark_replied('{lead['lead_id']}')\n"
        f"{'!'*60}\n"
    )

def _show_gui_popup(lead: dict):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showwarning(
        title="Missed Lead Reminder",
        message=(
            f"MISSED LEAD ALERT\n\n"
            f"Lead ID  : {lead['lead_id']}\n"
            f"Customer : {lead.get('customer_name', 'N/A')}\n"
            f"Channel  : {lead.get('channel', 'N/A')}\n"
            f"Gap      : {lead.get('gap_hrs', '?')} hrs without reply\n\n"
            f"Please follow up immediately!"
        ),
        parent=root
    )
    root.destroy()

def _reminder_loop(lead: dict, interval: int, stop_event: threading.Event):
    lead_id = lead["lead_id"]
    count   = 0
    while not stop_event.is_set():
        if is_replied(lead_id):
            print(f"[reminder] Lead {lead_id} replied - reminder thread exiting.")
            stop_event.set()
            break
        count += 1
        print(f"[reminder] Reminder #{count} for lead {lead_id}")
        if GUI_AVAILABLE:
            _show_gui_popup(lead)
        else:
            _print_banner(lead)
        stop_event.wait(timeout=interval)
    print(f"[reminder] Reminder loop for {lead_id} stopped.")

_active: dict = {}

def start_reminder(lead: dict, interval: int = INTERVAL_SECONDS):
    """Start repeating reminders. Safe to call multiple times (idempotent)."""
    lead_id = lead["lead_id"]
    if lead_id in _active and not _active[lead_id].is_set():
        print(f"[reminder] Reminder for {lead_id} already running.")
        return
    stop_ev = threading.Event()
    _active[lead_id] = stop_ev
    t = threading.Thread(target=_reminder_loop, args=(lead, interval, stop_ev),
                         daemon=True, name=f"reminder-{lead_id}")
    t.start()
    print(f"[reminder] Started reminder for lead {lead_id} (every {interval}s).")

def stop_reminder(lead_id: str):
    """Manually stop without marking replied."""
    if lead_id in _active:
        _active[lead_id].set()

if __name__ == "__main__":
    # Force terminal mode for demo (avoid GUI popups blocking the terminal)
    GUI_AVAILABLE = False

    sample_lead = {
        "lead_id"      : "L0042",
        "customer_name": "Priya",
        "channel"      : "WhatsApp",
        "gap_hrs"      : 36,
    }
    print("=== Employee Reminder Demo ===")
    start_reminder(sample_lead, interval=3)
    time.sleep(7)
    print("\n[DEMO] Employee replied - calling mark_replied()...")
    mark_replied(sample_lead["lead_id"])
    time.sleep(4)
    print("\n[DEMO] Complete - popup auto-dismissed.")
