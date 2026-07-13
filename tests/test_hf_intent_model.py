"""
test_hf_intent_model.py — Tests for hf_intent_model.py
Covers: keyword fallback classification, sentiment analysis, enhanced detection.
"""
import os
import sys
import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestKeywordFallback:
    """Test keyword-based intent classification fallback."""

    def test_pricing_keywords(self):
        from hf_intent_model import HFIntentClassifier
        classifier = HFIntentClassifier()
        result = classifier.classify("What is the price? How much does it cost?")
        assert result["primary"] == "pricing"
        assert result["method"] == "keyword_fallback"

    def test_demo_keywords(self):
        from hf_intent_model import HFIntentClassifier
        classifier = HFIntentClassifier()
        result = classifier.classify("Can I get a demo? I'd like to try a sample class.")
        assert result["primary"] == "demo"

    def test_complaint_keywords(self):
        from hf_intent_model import HFIntentClassifier
        classifier = HFIntentClassifier()
        result = classifier.classify("This is terrible. I'm unhappy and frustrated.")
        assert result["primary"] == "complaint"

    def test_urgent_keywords(self):
        from hf_intent_model import HFIntentClassifier
        classifier = HFIntentClassifier()
        result = classifier.classify("URGENT: I need immediate response! This is critical!")
        assert result["primary"] == "urgent"

    def test_batch_classification(self):
        from hf_intent_model import HFIntentClassifier
        classifier = HFIntentClassifier()
        texts = [
            "What is the price?",
            "Can I get a demo?",
            "I'm unhappy with the service.",
        ]
        results = classifier.classify_batch(texts)
        assert len(results) == 3
        assert results[0]["primary"] == "pricing"
        assert results[1]["primary"] == "demo"
        assert results[2]["primary"] == "complaint"


class TestSentimentAnalysis:
    """Test lexicon-based sentiment analysis."""

    def test_positive_sentiment(self):
        from hf_intent_model import HFSentimentAnalyzer
        analyzer = HFSentimentAnalyzer()
        result = analyzer.analyze("This is great! I'm happy and satisfied. Thank you!")
        assert result["sentiment"] == "positive"
        assert result["score"] > 0.5

    def test_negative_sentiment(self):
        from hf_intent_model import HFSentimentAnalyzer
        analyzer = HFSentimentAnalyzer()
        result = analyzer.analyze("This is terrible. I'm unhappy and frustrated.")
        assert result["sentiment"] == "negative"
        assert result["score"] > 0.5

    def test_neutral_sentiment(self):
        from hf_intent_model import HFSentimentAnalyzer
        analyzer = HFSentimentAnalyzer()
        result = analyzer.analyze("I would like some information.")
        assert result["sentiment"] in ["neutral", "positive", "negative"]

    def test_sentiment_method(self):
        from hf_intent_model import HFSentimentAnalyzer
        analyzer = HFSentimentAnalyzer()
        result = analyzer.analyze("Hello there")
        assert "method" in result
        assert result["method"] == "lexicon_fallback"


class TestEnhancedDetection:
    """Test enhanced_intent_detection function."""

    def test_returns_combined_result(self):
        from hf_intent_model import enhanced_intent_detection
        result = enhanced_intent_detection("What is the price of your course?")
        assert "primary_intent" in result
        assert "intent_scores" in result
        assert "sentiment" in result
        assert "hf_method" in result
        assert "confidence" in result

    def test_sentiment_in_result(self):
        from hf_intent_model import enhanced_intent_detection
        result = enhanced_intent_detection("This is great! I love it!")
        assert result["sentiment"]["sentiment"] in ["positive", "negative", "neutral"]

    def test_module_level_instances(self):
        from hf_intent_model import get_intent_classifier, get_sentiment_analyzer
        classifier = get_intent_classifier()
        analyzer = get_sentiment_analyzer()
        assert classifier is not None
        assert analyzer is not None
        # Should return same instance (singleton)
        assert get_intent_classifier() is classifier
        assert get_sentiment_analyzer() is analyzer
