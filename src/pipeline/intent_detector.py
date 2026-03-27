"""User intent detection for spatial tour guide follow-up questions.

Maps natural language input to a structured intent so the agent can route
to the right tool or response strategy — mirrors ZAUBAR's User Intent Agent.
"""

import logging
import re
from dataclasses import dataclass

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------

INTENT_NEARBY_PLACES = "nearby_places"
INTENT_HISTORICAL_FACTS = "historical_facts"
INTENT_TELL_MORE = "tell_me_more"
INTENT_OPENING_HOURS = "opening_hours"
INTENT_DIRECTIONS = "directions"
INTENT_TRANSLATION = "translation"
INTENT_PHOTO_TIP = "photo_tip"
INTENT_GENERAL = "general"

ALL_INTENTS = [
    INTENT_NEARBY_PLACES,
    INTENT_HISTORICAL_FACTS,
    INTENT_TELL_MORE,
    INTENT_OPENING_HOURS,
    INTENT_DIRECTIONS,
    INTENT_TRANSLATION,
    INTENT_PHOTO_TIP,
    INTENT_GENERAL,
]

# Keyword patterns per intent — used as fallback when no LLM key
_KEYWORD_PATTERNS: dict[str, list[str]] = {
    INTENT_NEARBY_PLACES: [
        r"\bnearby\b", r"\bclose by\b", r"\baround here\b", r"\bwhat else\b",
        r"\bother places\b", r"\bwhat can i visit\b", r"\bwhat('s| is) (around|near|close)\b",
        r"\bwalk(ing)? (distance|from here)\b", r"\bnext stop\b",
    ],
    INTENT_HISTORICAL_FACTS: [
        r"\bwhen was (it|this|the)\b", r"\bwho built\b", r"\bhistory\b",
        r"\bhow old\b", r"\bbuilt in\b", r"\bfounded\b", r"\boriginated\b",
        r"\barchitect\b", r"\bconstructed\b", r"\bwhat happened\b",
        r"\bhistorical(ly)?\b", r"\borigins?\b", r"\bpast\b",
    ],
    INTENT_TELL_MORE: [
        r"\btell me more\b", r"\bmore (details|info|information)\b",
        r"\belaborate\b", r"\bexpand\b", r"\bgo on\b", r"\bmore about\b",
        r"\bmore\?\s*$", r"\bkeep going\b", r"\bcontinue\b",
    ],
    INTENT_OPENING_HOURS: [
        r"\bopen(ing)? hours\b", r"\bwhen (does it open|is it open|do they open)\b",
        r"\bclosed\b", r"\bticket(s|ing)?\b", r"\badmission\b",
        r"\bhow much (does it cost|to enter|is entry)\b",
        r"\bentr(y|ance) fee\b", r"\bhours?\b",
    ],
    INTENT_DIRECTIONS: [
        r"\bhow (do i|can i|to) get (there|to)\b", r"\bdirection(s)?\b",
        r"\bnavigate\b", r"\bwalk(ing)? (route|here|there)\b",
        r"\bpublic transport\b", r"\bbus|u-bahn|s-bahn|subway|metro\b",
        r"\bhow far\b", r"\bdistance\b",
    ],
    INTENT_TRANSLATION: [
        r"\bsay (that|this) in\b", r"\btranslate\b", r"\bin (german|french|spanish|"
        r"italian|dutch|polish|portuguese|chinese|japanese)\b",
        r"\bwie (sagt|heißt)\b", r"\bauf deutsch\b",
    ],
    INTENT_PHOTO_TIP: [
        r"\bphoto(graph)?\b", r"\bpicture\b", r"\bshot\b", r"\bangle\b",
        r"\blight(ing)?\b", r"\bbest time (to|for)\b",
        r"\bhow (should|to) (i |we )?photo\b",
    ],
}


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: str
    confidence: float  # 0.0–1.0 (1.0 = keyword match, 0.0–1.0 from LLM)
    method: str        # "llm" | "keyword" | "default"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class IntentDetector:
    """Classify user questions into structured intents.

    Uses GPT-4o-mini when OPENAI_API_KEY is configured.
    Falls back to keyword matching when no API key is available.
    """

    _SYSTEM_PROMPT = (
        "You are an intent classifier for a spatial tour guide app. "
        "Classify the user's question into exactly one of these intents:\n"
        f"  {', '.join(ALL_INTENTS)}\n\n"
        "Definitions:\n"
        "  nearby_places   — asking what else is near, what to visit next\n"
        "  historical_facts — asking about history, age, architect, events\n"
        "  tell_me_more    — asking for elaboration on what was just said\n"
        "  opening_hours   — asking about hours, tickets, entry fee\n"
        "  directions      — asking how to get somewhere or transit options\n"
        "  translation     — asking to translate or say something in another language\n"
        "  photo_tip       — asking for photography advice at this spot\n"
        "  general         — anything else\n\n"
        "Reply with ONLY the intent name, nothing else."
    )

    def detect(self, question: str, session_context: dict | None = None) -> IntentResult:
        """Detect the intent of a follow-up question.

        Args:
            question: The user's question text.
            session_context: Optional dict with 'landmark_name', 'scene' for context.

        Returns:
            IntentResult with intent, confidence, and method used.
        """
        if settings.openai_api_key:
            try:
                return self._detect_with_llm(question)
            except Exception as exc:
                logger.warning("LLM intent detection failed, falling back: %s", exc)
        return self._detect_with_keywords(question)

    def _detect_with_llm(self, question: str) -> IntentResult:
        """Use GPT-4o-mini for fast, accurate intent classification."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.gpt_model,
            api_key=settings.openai_api_key,
            max_tokens=10,
            temperature=0,
        )
        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
        raw = llm.invoke(messages).content.strip().lower()
        intent = raw if raw in ALL_INTENTS else INTENT_GENERAL
        logger.info("IntentDetector(llm): %r → %s", question[:60], intent)
        return IntentResult(intent=intent, confidence=0.9, method="llm")

    def _detect_with_keywords(self, question: str) -> IntentResult:
        """Rule-based keyword matching fallback."""
        q = question.lower()
        for intent, patterns in _KEYWORD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q):
                    logger.info("IntentDetector(keyword): %r → %s", question[:60], intent)
                    return IntentResult(intent=intent, confidence=1.0, method="keyword")
        logger.info("IntentDetector(keyword): %r → general (no match)", question[:60])
        return IntentResult(intent=INTENT_GENERAL, confidence=0.5, method="default")
