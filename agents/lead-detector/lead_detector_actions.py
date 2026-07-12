import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import orchestrator
import smart_reply_engine


def _infer_lead_from_text(text: str) -> dict:
    if not text:
        return {}

    text = str(text).strip()
    lead = {"message_text": text}

    name_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    if name_match:
        lead["name"] = name_match.group(1)

    company_match = re.search(r"\bfrom\s+([A-Za-z0-9&.,' -]+?)(?:,|$)", text, re.I)
    if company_match:
        lead["company"] = company_match.group(1).strip()

    days_match = re.search(r"(?:no reply|no response|reply|replied|responded|follow-up|follow up|hasn't replied|haven't replied|not replied)[^\d]*(\d+)\s*(day|days)", text, re.I)
    if days_match:
        lead["days_since_contact"] = float(days_match.group(1))

    if "last message" in text.lower():
        msg_match = re.search(r"last message(?:\s+was)?\s*[\"']([^\"']+)[\"']", text, re.I)
        if msg_match:
            lead["last_message"] = msg_match.group(1)

    if "last message" not in text.lower() and "message" in text.lower():
        msg_match = re.search(r"(?:message|said)\s*[\"']([^\"']+)[\"']", text, re.I)
        if msg_match:
            lead["last_message"] = msg_match.group(1)

    if "last_message" not in lead:
        lead["last_message"] = lead.get("message_text", "")

    return lead


def _coerce_lead(lead_payload: dict | str) -> dict:
    if isinstance(lead_payload, str):
        lead = _infer_lead_from_text(lead_payload)
    else:
        lead = dict(lead_payload or {})
        message_text = (
            lead.get("message_text")
            or lead.get("message")
            or lead.get("last_message")
            or ""
        )
        inferred = _infer_lead_from_text(message_text)
        for key in ["name", "company", "days_since_contact", "last_message", "message_text"]:
            if not lead.get(key) and inferred.get(key):
                lead[key] = inferred[key]

    lead_name = (
        lead.get("name")
        or lead.get("lead_name")
        or lead.get("customer_name")
        or lead.get("contact_name")
        or "Unknown"
    )
    company = (
        lead.get("company")
        or lead.get("organization")
        or lead.get("business")
        or ""
    )
    last_message = (
        lead.get("last_message")
        or lead.get("message")
        or lead.get("message_text")
        or ""
    )
    channel = str(lead.get("channel") or lead.get("source") or "email")
    days_since_contact = lead.get("days_since_contact")
    if days_since_contact is None:
        days_since_contact = lead.get("days_since_last_contact")
    if days_since_contact is None:
        days_since_contact = lead.get("response_gap_days")
    if days_since_contact is None:
        days_since_contact = lead.get("response_gap_hrs")
    if days_since_contact is None:
        days_since_contact = 0

    try:
        days_since_contact = float(days_since_contact)
    except (TypeError, ValueError):
        days_since_contact = 0.0

    prev_contacts = lead.get("prev_contacts")
    if prev_contacts is None:
        prev_contacts = lead.get("previous_contacts")
    if prev_contacts is None:
        prev_contacts = lead.get("follow_up_count")
    if prev_contacts is None:
        prev_contacts = 0
    try:
        prev_contacts = int(prev_contacts)
    except (TypeError, ValueError):
        prev_contacts = 0

    message_hour = lead.get("message_hour")
    if message_hour is None:
        message_hour = lead.get("contact_hour")
    if message_hour is None:
        message_hour = 12
    try:
        message_hour = int(message_hour)
    except (TypeError, ValueError):
        message_hour = 12

    text = str(last_message or "")
    intent = smart_reply_engine.detect_intent(text)
    high_intent_flag = lead.get("high_intent_flag")
    if high_intent_flag is None:
        high_intent_flag = 1 if intent["primary"] in {"pricing", "demo", "course", "placement", "availability", "urgent"} else 0
    try:
        high_intent_flag = int(high_intent_flag)
    except (TypeError, ValueError):
        high_intent_flag = 0

    lead = {
        "lead_id": lead.get("lead_id") or f"{lead_name}-{company}".strip("-") or "lead-1",
        "name": lead_name,
        "company": company,
        "channel": channel,
        "message_text": text,
        "message_length": len(text),
        "high_intent_flag": high_intent_flag,
        "prev_contacts": prev_contacts,
        "response_gap_hrs": max(0.0, days_since_contact * 24.0),
        "message_hour": message_hour,
    }
    return lead


def detect_intent(message_text: str) -> dict:
    return smart_reply_engine.detect_intent(message_text)


def draft_reply(customer_name: str, customer_email: str, subject: str, message_text: str, channel: str = "Email") -> dict:
    return smart_reply_engine.generate_reply(
        customer_name=customer_name,
        customer_email=customer_email,
        subject=subject,
        message_text=message_text,
        channel=channel,
    )


def score_lead(lead_payload: dict | str) -> dict:
    lead = _coerce_lead(lead_payload)
    df = pd.DataFrame([lead])

    probability = 0.0
    try:
        ensemble, scaler, dl_model, dl_scaler = orchestrator.load_artefacts()
        scored_df = orchestrator.score_leads(df.copy(), ensemble, scaler, dl_model, dl_scaler)
        probability = float(scored_df.iloc[0].get("missed_probability", 0.0))
    except Exception:
        probability = 0.3 + 0.2 * (1 - lead.get("high_intent_flag", 0)) + 0.1 * min(lead.get("response_gap_hrs", 0.0) / 100.0, 0.5)

    text = str(lead.get("message_text", "")).lower()
    days_since_contact = max(0.0, lead.get("response_gap_hrs", 0.0) / 24.0)

    rubric_score = 3.0
    if days_since_contact >= 10:
        rubric_score += 2.5
    elif days_since_contact >= 5:
        rubric_score += 1.8
    elif days_since_contact >= 3:
        rubric_score += 1.0
    elif days_since_contact >= 1:
        rubric_score += 0.5

    if any(token in text for token in ["send more info", "more info", "interested", "sounds interesting", "would like", "demo", "learn more"]):
        rubric_score += 2.5
    elif any(token in text for token in ["not interested", "not now", "no thanks", "not right now", "stop", "no longer interested"]):
        rubric_score -= 3.5
    elif any(token in text for token in ["let me think", "maybe", "consider", "think about it", "next quarter"]):
        rubric_score += 0.5

    if int(lead.get("prev_contacts", 0)) > 0:
        rubric_score += 0.5

    if any(token in text for token in ["urgent", "immediately", "asap"]):
        rubric_score += 1.0

    rubric_score = max(0.0, min(10.0, round(rubric_score, 1)))
    model_score = probability * 10.0
    score = round((rubric_score * 0.7) + (model_score * 0.3), 1)
    score = max(0.0, min(10.0, score))

    if score >= 8.0:
        priority = "Hot"
        reasoning = "The lead shows strong engagement or a long silence, so it is worth prioritizing now."
    elif score >= 5.0:
        priority = "Warm"
        reasoning = "The lead has moderate interest and some delay, so a timely follow-up is sensible."
    else:
        priority = "Cold"
        reasoning = "The lead looks low-signal or explicitly disengaged, so it should be handled last."

    intent = smart_reply_engine.detect_intent(str(lead.get("message_text", "")))
    return {
        "lead_id": lead.get("lead_id", ""),
        "name": lead.get("name", ""),
        "company": lead.get("company", ""),
        "channel": lead.get("channel", "email"),
        "score": score,
        "priority": priority,
        "reasoning": reasoning,
        "predicted_missed": int(probability >= 0.5),
        "missed_probability": round(probability, 4),
        "intent": intent.get("primary", "interest"),
        "intent_scores": intent.get("scores", {}),
        "days_since_contact": round(days_since_contact, 1),
        "last_message": lead.get("message_text", ""),
    }


def _parse_lead_payload(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}

    if text.startswith("{") and text.endswith("}"):
        text = text.replace("'", '"')
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    def _parse_simple_object(obj_text: str) -> dict:
        obj_text = obj_text.strip()
        if not obj_text.startswith("{") or not obj_text.endswith("}"):
            return {}

        inner = obj_text[1:-1].strip()
        if not inner:
            return {}

        result = {}
        parts = []
        current = []
        in_string = False
        quote = None
        for char in inner:
            if in_string:
                current.append(char)
                if char == quote:
                    in_string = False
                    quote = None
                continue
            if char in {'"', "'"}:
                in_string = True
                quote = char
                current.append(char)
            elif char == ',':
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append(''.join(current).strip())

        for part in parts:
            if ':' not in part:
                continue
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key.startswith(("'", '"')) and key.endswith(("'", '"')):
                key = key[1:-1]
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            try:
                if value.lower() in {"true", "false"}:
                    value = value.lower() == "true"
                elif value.lower() in {"null", "none"}:
                    value = None
                elif value.replace('.', '', 1).isdigit():
                    value = int(value) if '.' not in value else float(value)
            except Exception:
                pass
            result[key] = value
        return result

    parsed_simple = _parse_simple_object(text)
    if parsed_simple:
        return parsed_simple

    candidates = [text]
    if text and text[0] == text[-1] and text[0] in {"'", '"'}:
        candidate = text[1:-1]
        if candidate:
            candidates.append(candidate)

    for item in candidates:
        try:
            parsed = json.loads(item)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(item)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                inner = parsed.strip()
                if inner:
                    try:
                        parsed_inner = json.loads(inner)
                        if isinstance(parsed_inner, dict):
                            return parsed_inner
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass

    raise ValueError(f"Unable to parse lead payload: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead detector wrapper actions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect-intent")
    detect_parser.add_argument("--message", required=True)

    draft_parser = subparsers.add_parser("draft-reply")
    draft_parser.add_argument("--customer-name", default="there")
    draft_parser.add_argument("--customer-email", default="customer@example.com")
    draft_parser.add_argument("--subject", default="Your inquiry")
    draft_parser.add_argument("--message", required=True)
    draft_parser.add_argument("--channel", default="Email")

    score_parser = subparsers.add_parser("score-lead")
    score_parser.add_argument("--lead-json", default=None)
    score_parser.add_argument("--lead-text", default=None)
    score_parser.add_argument("--name")
    score_parser.add_argument("--company")
    score_parser.add_argument("--days-since-contact", type=float)
    score_parser.add_argument("--prev-contacts", type=int)
    score_parser.add_argument("--last-message")
    score_parser.add_argument("--channel")
    score_parser.add_argument("--message-hour", type=int)

    args = parser.parse_args()

    if args.command == "detect-intent":
        print(json.dumps(detect_intent(args.message), indent=2))
    elif args.command == "draft-reply":
        print(json.dumps(draft_reply(
            customer_name=args.customer_name,
            customer_email=args.customer_email,
            subject=args.subject,
            message_text=args.message,
            channel=args.channel,
        ), indent=2))
    elif args.command == "score-lead":
        payload = {}
        if args.lead_json:
            payload.update(_parse_lead_payload(args.lead_json))
        if args.lead_text:
            payload = {"message_text": args.lead_text}
        if args.name is not None:
            payload["name"] = args.name
        if args.company is not None:
            payload["company"] = args.company
        if args.days_since_contact is not None:
            payload["days_since_contact"] = args.days_since_contact
        if args.prev_contacts is not None:
            payload["prev_contacts"] = args.prev_contacts
        if args.last_message is not None:
            payload["last_message"] = args.last_message
        if args.channel is not None:
            payload["channel"] = args.channel
        if args.message_hour is not None:
            payload["message_hour"] = args.message_hour
        print(json.dumps(score_lead(payload), indent=2))


if __name__ == "__main__":
    main()
