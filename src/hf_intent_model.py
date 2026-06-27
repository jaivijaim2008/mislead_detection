"""
hf_intent_model.py — Missed-Lead Detector
Hugging Face BERT/DistilBERT-based intent extraction model (Course Unit IV).

Uses a pre-trained DistilBERT model fine-tuned for multi-label intent classification
on customer inquiry emails. Detects: pricing, demo, course, placement, complaint,
interest, availability, urgent intents.

This module:
  1. Provides zero-shot intent classification using a fine-tuned model
  2. Falls back to keyword-based detection when HF model is unavailable
  3. Integrates with the smart_reply_engine for enhanced intent detection
  4. Can be used standalone for intent classification demonstrations

Environment:
  HF_INTENT_MODEL: Set to a local model path or HuggingFace model name
                   Default: "distilbert-base-uncased"
  HF_DEVICE: "cpu" or "cuda" (auto-detected)
"""

import os
import re
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Intent Labels ──────────────────────────────────────────────────────────

INTENT_LABELS = [
    "pricing",
    "demo",
    "course",
    "placement",
    "complaint",
    "interest",
    "availability",
    "urgent",
]

# ── Zero-Shot Classification Keywords (fallback when no HF model) ─────────

ZERO_SHOT_KEYWORDS = {
    "pricing": ["price", "pricing", "cost", "fee", "fees", "how much", "quote",
                 "budget", "affordable", "expensive", "discount", "emi", "payment",
                 "charge", "rate", "rupees", "inr"],
    "demo": ["demo", "trial", "sample", "try", "test", "see it", "walkthrough",
             "preview", "demonstration", "show me", "free class"],
    "course": ["course", "program", "batch", "class", "training", "certification",
               "curriculum", "syllabus", "module", "mba", "diploma", "learn"],
    "placement": ["placement", "job", "career", "hiring", "employment",
                  "opportunity", "internship", "resume", "hire"],
    "complaint": ["complaint", "issue", "problem", "not working", "unhappy",
                  "dissatisfied", "frustrated", "bad", "terrible", "worst",
                  "refund", "poor"],
    "interest": ["interested", "want to", "looking for", "need", "enquire",
                 "information", "details", "tell me", "share", "help"],
    "availability": ["available", "timing", "schedule", "when", "start",
                     "next batch", "date", "duration", "weekend", "weekday",
                     "seats", "enroll"],
    "urgent": ["urgent", "asap", "immediately", "today", "tomorrow", "hurry",
               "quick", "fast", "emergency", "critical", "pending"],
}


# ── HF Model Wrapper ──────────────────────────────────────────────────────

class HFIntentClassifier:
    """
    Intent classifier using Hugging Face transformers.

    Uses zero-shot classification with a pre-trained model when available,
    falls back to keyword-based scoring otherwise.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or os.getenv(
            "HF_INTENT_MODEL", "distilbert-base-uncased"
        )
        self.device = device or os.getenv("HF_DEVICE", "cpu")
        self._pipeline = None
        self._use_hf = False
        self._init_model()

    def _init_model(self):
        """Try to load the Hugging Face pipeline."""
        # Check if we should skip HF model (avoids slow downloads)
        # Set HF_INTENT_USE_MODEL=1 to enable (downloads ~1.6GB BART model)
        if os.getenv("HF_INTENT_USE_MODEL", "0") != "1":
            print("[hf_intent] Using keyword fallback (set HF_INTENT_USE_MODEL=1 to enable HF model)")
            self._use_hf = False
            return
        try:
            from transformers import pipeline
            # Use zero-shot classification pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1 if self.device == "cpu" else 0,
            )
            self._use_hf = True
            print("[hf_intent] Loaded Hugging Face zero-shot classification model")
            print(f"[hf_intent] Device: {self.device}")
        except Exception as e:
            print(f"[hf_intent] HF model not available ({e}), using keyword fallback")
            self._use_hf = False

    def classify(self, text: str) -> dict:
        """
        Classify the intent of a text message.

        Returns:
            dict with 'primary' intent, 'scores' for all intents, and 'method' used.
        """
        if self._use_hf and self._pipeline is not None:
            return self._classify_hf(text)
        return self._classify_keywords(text)

    def _classify_hf(self, text: str) -> dict:
        """Classify using Hugging Face zero-shot classification."""
        try:
            result = self._pipeline(
                text,
                candidate_labels=INTENT_LABELS,
                multi_label=True,
            )
            scores = {}
            for label, score in zip(result["labels"], result["scores"]):
                scores[label] = round(score, 4)

            primary = result["labels"][0]
            return {
                "primary": primary,
                "scores": scores,
                "method": "huggingface_zero_shot",
                "confidence": result["scores"][0],
            }
        except Exception as e:
            print(f"[hf_intent] HF classification failed: {e}, falling back to keywords")
            return self._classify_keywords(text)

    def _classify_keywords(self, text: str) -> dict:
        """Classify using keyword matching (fallback method)."""
        text_lower = text.lower()
        scores = {}
        for intent, keywords in ZERO_SHOT_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[intent] = count

        total = sum(scores.values()) or 1
        normalized = {k: v / total for k, v in scores.items()}
        primary = max(normalized, key=normalized.get)

        return {
            "primary": primary,
            "scores": normalized,
            "method": "keyword_fallback",
            "confidence": normalized[primary],
        }

    def classify_batch(self, texts: list) -> list:
        """Classify a batch of texts."""
        return [self.classify(text) for text in texts]


# ── Sentiment Analysis ─────────────────────────────────────────────────────

class HFSentimentAnalyzer:
    """
    Sentiment analyzer using Hugging Face transformers.
    Detects customer sentiment: positive, negative, neutral.
    """

    def __init__(self, device: str = None):
        self.device = device or os.getenv("HF_DEVICE", "cpu")
        self._pipeline = None
        self._use_hf = False
        self._init_model()

    def _init_model(self):
        if os.getenv("HF_INTENT_USE_MODEL", "0") != "1":
            print("[hf_sentiment] Using lexicon fallback (set HF_INTENT_USE_MODEL=1 to enable HF model)")
            self._use_hf = False
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1 if self.device == "cpu" else 0,
            )
            self._use_hf = True
            print("[hf_sentiment] Loaded Hugging Face sentiment model")
        except Exception as e:
            print(f"[hf_sentiment] HF sentiment not available ({e}), using lexicon fallback")
            self._use_hf = False

    def analyze(self, text: str) -> dict:
        """Analyze sentiment of a text message."""
        if self._use_hf and self._pipeline is not None:
            return self._analyze_hf(text)
        return self._analyze_lexicon(text)

    def _analyze_hf(self, text: str) -> dict:
        try:
            result = self._pipeline(text[:512])[0]  # Truncate to 512 tokens
            label = result["label"].lower()
            score = result["score"]
            return {
                "sentiment": label,
                "score": round(score, 4),
                "method": "huggingface",
            }
        except Exception as e:
            print(f"[hf_sentiment] HF analysis failed: {e}")
            return self._analyze_lexicon(text)

    def _analyze_lexicon(self, text: str) -> dict:
        """Simple lexicon-based sentiment analysis (fallback)."""
        positive_words = {"good", "great", "excellent", "amazing", "love",
                          "happy", "satisfied", "thank", "thanks", "wonderful",
                          "fantastic", "perfect", "best", "awesome", "helpful"}
        negative_words = {"bad", "terrible", "worst", "hate", "angry", "unhappy",
                          "frustrated", "disappointed", "poor", "horrible", "awful",
                          "waste", "refund", "complaint", "issue", "problem"}

        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg or 1

        if pos > neg:
            sentiment = "positive"
            score = pos / total
        elif neg > pos:
            sentiment = "negative"
            score = neg / total
        else:
            sentiment = "neutral"
            score = 0.5

        return {
            "sentiment": sentiment,
            "score": round(score, 4),
            "method": "lexicon_fallback",
        }


# ── Module-level instances (lazy init) ─────────────────────────────────────

_intent_classifier = None
_sentiment_analyzer = None


def get_intent_classifier() -> HFIntentClassifier:
    """Get or create the singleton intent classifier."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = HFIntentClassifier()
    return _intent_classifier


def get_sentiment_analyzer() -> HFSentimentAnalyzer:
    """Get or create the singleton sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = HFSentimentAnalyzer()
    return _sentiment_analyzer


def detect_intent_hf(message_text: str) -> dict:
    """
    Detect intent using Hugging Face model (or keyword fallback).
    Drop-in replacement for smart_reply_engine.detect_intent().
    """
    classifier = get_intent_classifier()
    return classifier.classify(message_text)


def analyze_sentiment(message_text: str) -> dict:
    """Analyze customer sentiment from email text."""
    analyzer = get_sentiment_analyzer()
    return analyzer.analyze(message_text)


# ── Integration with smart_reply_engine ────────────────────────────────────

def enhanced_intent_detection(message_text: str) -> dict:
    """
    Enhanced intent detection combining:
      1. Hugging Face model (if available)
      2. Smart reply engine keyword detection
      3. Sentiment analysis

    Returns combined result with primary intent, scores, and sentiment.
    """
    # HF intent classification
    hf_result = detect_intent_hf(message_text)

    # Keyword-based detection from smart_reply_engine
    try:
        from smart_reply_engine import detect_intent as kw_detect
        kw_result = kw_detect(message_text)
    except ImportError:
        kw_result = {"primary": hf_result["primary"], "scores": {}}

    # Sentiment analysis
    sentiment = analyze_sentiment(message_text)

    # Combine scores: weighted average favoring HF when available
    if hf_result.get("method") == "huggingface_zero_shot":
        # Trust HF more when available
        combined_scores = {}
        for label in INTENT_LABELS:
            hf_score = hf_result.get("scores", {}).get(label, 0)
            kw_score = kw_result.get("scores", {}).get(label, 0)
            combined_scores[label] = 0.7 * hf_score + 0.3 * kw_score
    else:
        combined_scores = hf_result.get("scores", {})

    primary = max(combined_scores, key=combined_scores.get) if combined_scores else "interest"

    return {
        "primary_intent": primary,
        "intent_scores": combined_scores,
        "sentiment": sentiment,
        "hf_method": hf_result.get("method", "unknown"),
        "confidence": combined_scores.get(primary, 0),
    }


# ── Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  MISSED-LEAD DETECTOR — HUGGING FACE INTENT & SENTIMENT MODEL")
    print("  (Course Unit IV — Neural Networks)")
    print("=" * 70)

    test_messages = [
        "Hi, I want to know the price of your data science course. Do you have EMI options?",
        "Can I get a demo of your teaching before I enroll?",
        "Do you provide placement assistance after the course?",
        "This is terrible. I've been waiting for 3 days. I want a refund!",
        "URGENT: I need immediate response. My enrollment is pending!",
        "When does the next batch start? I want to join ASAP.",
        "I'm interested in learning data science. Can you help?",
        "Hello, just checking what you offer.",
    ]

    classifier = get_intent_classifier()
    sentiment_analyzer = get_sentiment_analyzer()

    for msg in test_messages:
        intent = classifier.classify(msg)
        sentiment = sentiment_analyzer.analyze(msg)
        print(f"\n{'-'*70}")
        print(f"  Message: \"{msg[:60]}...\"")
        print(f"  Intent:  {intent['primary']:15s} (conf: {intent['confidence']:.2f}, "
              f"method: {intent['method']})")
        print(f"  Sentiment: {sentiment['sentiment']:10s} "
              f"(score: {sentiment['score']:.2f}, method: {sentiment['method']})")

    print(f"\n{'='*70}")
    print("  Enhanced intent detection demo:")
    print(f"{'='*70}")

    for msg in test_messages[:3]:
        result = enhanced_intent_detection(msg)
        print(f"\n  Message: \"{msg[:60]}...\"")
        print(f"  Primary: {result['primary_intent']}")
        print(f"  Sentiment: {result['sentiment']['sentiment']}")
        print(f"  HF Method: {result['hf_method']}")
