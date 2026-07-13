"""
test_smart_reply_engine.py — Tests for smart_reply_engine.py
Covers: intent detection, template selection, reply generation, placeholder filling.
"""
import os
import sys
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestIntentDetection:
    """Test detect_intent function."""

    def test_pricing_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("Hi, I want to know the price of your course. How much does it cost?")
        assert result["primary"] == "pricing"
        assert "pricing" in result["scores"]

    def test_demo_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("Can I get a demo of your teaching? I'd like to try a sample class.")
        assert result["primary"] == "demo"

    def test_course_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("Tell me about your data science curriculum and batch schedule.")
        assert result["primary"] == "course"

    def test_placement_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("Do you provide placement assistance? What is your placement rate?")
        assert result["primary"] == "placement"

    def test_complaint_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("I'm unhappy with the service. This is terrible and frustrating.")
        assert result["primary"] == "complaint"

    def test_interest_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("I'm interested in learning data science. Can you help me?")
        assert result["primary"] == "interest"

    def test_availability_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("When does the next batch start? What are the available slots?")
        assert result["primary"] == "availability"

    def test_urgent_intent_detected(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("URGENT: I need immediate response. This is critical!")
        assert result["primary"] == "urgent"

    def test_scores_are_normalized(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("price and demo please")
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 0.01, f"Scores don't sum to 1: {total}"

    def test_all_intents_present_in_scores(self):
        from smart_reply_engine import detect_intent
        expected_intents = {"pricing", "demo", "course", "placement",
                           "complaint", "interest", "availability", "urgent"}
        result = detect_intent("hello there")
        assert set(result["scores"].keys()) == expected_intents

    def test_empty_text_handling(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("")
        assert result["primary"] in ["pricing", "demo", "course", "placement",
                                     "complaint", "interest", "availability", "urgent"]

    def test_fallback_for_no_keywords(self):
        from smart_reply_engine import detect_intent
        result = detect_intent("just browsing")
        assert result["primary"] is not None


class TestReplyGeneration:
    """Test generate_reply function."""

    def test_returns_required_keys(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Priya",
            customer_email="priya@example.com",
            subject="Course Pricing",
            message_text="Hi, what is the price?",
            channel="Email",
        )
        assert "reply_subject" in reply
        assert "reply_body" in reply
        assert "detected_intent" in reply
        assert "intent_scores" in reply
        assert "channel" in reply
        assert "is_auto_replied" in reply
        assert "generated_at" in reply

    def test_reply_subject_contains_re(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Rahul",
            customer_email="rahul@example.com",
            subject="Course Info",
            message_text="Tell me about your courses",
            channel="Email",
        )
        assert "Re:" in reply["reply_subject"]

    def test_reply_body_contains_customer_name(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Priya",
            customer_email="priya@example.com",
            subject="Pricing",
            message_text="What is the price?",
            channel="Email",
        )
        assert "Priya" in reply["reply_body"]

    def test_reply_body_contains_sender_name(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Test",
            customer_email="test@example.com",
            subject="Hi",
            message_text="hello",
            channel="Email",
        )
        from config import SENDER_NAME
        assert SENDER_NAME in reply["reply_body"]

    def test_reply_body_contains_phone(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Test",
            customer_email="test@example.com",
            subject="Help",
            message_text="I need help with pricing",
            channel="Email",
        )
        from config import TEAM_PHONE
        assert TEAM_PHONE in reply["reply_body"]

    def test_is_auto_replied_flag(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Test",
            customer_email="test@example.com",
            subject="Hi",
            message_text="hello",
            channel="Email",
        )
        assert reply["is_auto_replied"] is True

    def test_channel_preserved(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Test",
            customer_email="test@example.com",
            subject="Hi",
            message_text="hello",
            channel="WhatsApp",
        )
        assert reply["channel"] == "WhatsApp"

    def test_no_auto_generated_footer(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Test",
            customer_email="test@example.com",
            subject="Hi",
            message_text="hello",
            channel="Email",
        )
        assert "Auto-generated" not in reply["reply_body"]

    def test_pricing_reply_contains_emi(self):
        from smart_reply_engine import generate_reply
        reply = generate_reply(
            customer_name="Priya",
            customer_email="priya@example.com",
            subject="Pricing",
            message_text="What is the price? How much? EMI options?",
            channel="Email",
        )
        # Pricing intent should mention EMI or pricing details
        body_lower = reply["reply_body"].lower()
        assert "emi" in body_lower or "price" in body_lower or "fee" in body_lower


class TestFormatReplyPreview:
    """Test format_reply_preview function."""

    def test_returns_string(self):
        from smart_reply_engine import format_reply_preview
        preview = format_reply_preview(
            customer_name="Priya",
            customer_email="priya@example.com",
            subject="Pricing",
            message_text="What is the price?",
        )
        assert isinstance(preview, str)

    def test_contains_preview_header(self):
        from smart_reply_engine import format_reply_preview
        preview = format_reply_preview(
            customer_name="Priya",
            customer_email="priya@example.com",
            subject="Pricing",
            message_text="What is the price?",
        )
        assert "AUTO-REPLY PREVIEW" in preview
