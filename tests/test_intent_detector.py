"""Tests for IntentDetector — keyword and LLM paths."""

from unittest.mock import patch

import pytest

from src.pipeline.intent_detector import (
    INTENT_DIRECTIONS,
    INTENT_GENERAL,
    INTENT_HISTORICAL_FACTS,
    INTENT_NEARBY_PLACES,
    INTENT_OPENING_HOURS,
    INTENT_PHOTO_TIP,
    INTENT_TELL_MORE,
    INTENT_TRANSLATION,
    IntentDetector,
)

# ---------------------------------------------------------------------------
# Keyword fallback tests (no LLM)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_openai_key():
    """Force keyword-based detection for all tests by default."""
    with patch("src.pipeline.intent_detector.settings") as mock_settings:
        mock_settings.openai_api_key = ""
        yield mock_settings


class TestKeywordDetection:
    def _detect(self, question: str):
        return IntentDetector().detect(question)

    def test_nearby_places_nearby(self):
        r = self._detect("What else is nearby?")
        assert r.intent == INTENT_NEARBY_PLACES
        assert r.method == "keyword"

    def test_nearby_places_visit(self):
        r = self._detect("What can I visit close by?")
        assert r.intent == INTENT_NEARBY_PLACES

    def test_nearby_places_what_else(self):
        r = self._detect("What else can I see around here?")
        assert r.intent == INTENT_NEARBY_PLACES

    def test_historical_facts_when_built(self):
        r = self._detect("When was it built?")
        assert r.intent == INTENT_HISTORICAL_FACTS

    def test_historical_facts_who_built(self):
        r = self._detect("Who built this monument?")
        assert r.intent == INTENT_HISTORICAL_FACTS

    def test_historical_facts_keyword(self):
        r = self._detect("Tell me about its history.")
        assert r.intent == INTENT_HISTORICAL_FACTS

    def test_tell_me_more(self):
        r = self._detect("Tell me more about this place!")
        assert r.intent == INTENT_TELL_MORE

    def test_tell_me_more_elaborate(self):
        r = self._detect("Can you elaborate on that?")
        assert r.intent == INTENT_TELL_MORE

    def test_opening_hours(self):
        r = self._detect("What are the opening hours?")
        assert r.intent == INTENT_OPENING_HOURS

    def test_opening_hours_tickets(self):
        r = self._detect("How much are tickets?")
        assert r.intent == INTENT_OPENING_HOURS

    def test_directions(self):
        r = self._detect("How do I get there by public transport?")
        assert r.intent == INTENT_DIRECTIONS

    def test_directions_u_bahn(self):
        r = self._detect("Which U-Bahn station is closest?")
        assert r.intent == INTENT_DIRECTIONS

    def test_translation(self):
        r = self._detect("How do you say that in German?")
        assert r.intent == INTENT_TRANSLATION

    def test_translation_auf_deutsch(self):
        r = self._detect("Auf Deutsch bitte?")
        assert r.intent == INTENT_TRANSLATION

    def test_photo_tip(self):
        r = self._detect("What's the best angle for a photo here?")
        assert r.intent == INTENT_PHOTO_TIP

    def test_general_fallback(self):
        r = self._detect("Interesting!")
        assert r.intent == INTENT_GENERAL
        assert r.method == "default"

    def test_general_random(self):
        r = self._detect("That's amazing, thanks.")
        assert r.intent == INTENT_GENERAL

    def test_confidence_keyword_is_1(self):
        r = self._detect("What's nearby?")
        assert r.confidence == 1.0

    def test_confidence_default_is_0_5(self):
        r = self._detect("Wow!")
        assert r.confidence == 0.5


# ---------------------------------------------------------------------------
# LLM path tests
# ---------------------------------------------------------------------------

class TestLLMDetection:
    # ChatOpenAI is imported locally inside _detect_with_llm — patch there
    _LLM_PATH = "src.pipeline.intent_detector.IntentDetector._detect_with_llm"

    def test_llm_path_called_when_key_set(self):
        from src.pipeline.intent_detector import IntentResult

        mock_result = IntentResult(intent=INTENT_NEARBY_PLACES, confidence=0.9, method="llm")

        with patch("src.pipeline.intent_detector.settings") as mock_settings, \
             patch(self._LLM_PATH, return_value=mock_result):
            mock_settings.openai_api_key = "sk-test"
            result = IntentDetector().detect("What's close by?")

        assert result.intent == INTENT_NEARBY_PLACES
        assert result.method == "llm"
        assert result.confidence == 0.9

    def test_llm_unknown_intent_falls_back_to_general(self):
        from src.pipeline.intent_detector import IntentResult

        mock_result = IntentResult(intent=INTENT_GENERAL, confidence=0.9, method="llm")

        with patch("src.pipeline.intent_detector.settings") as mock_settings, \
             patch(self._LLM_PATH, return_value=mock_result):
            mock_settings.openai_api_key = "sk-test"
            result = IntentDetector().detect("Some question")

        assert result.intent == INTENT_GENERAL

    def test_llm_exception_falls_back_to_keyword(self):
        with patch("src.pipeline.intent_detector.settings") as mock_settings, \
             patch(self._LLM_PATH, side_effect=RuntimeError("API down")):
            mock_settings.openai_api_key = "sk-test"
            result = IntentDetector().detect("What's nearby?")

        # Falls back to keyword → nearby_places
        assert result.intent == INTENT_NEARBY_PLACES
        assert result.method == "keyword"
