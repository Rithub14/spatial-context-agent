"""Narration engine — Claude-powered with RAG, falls back to templates if no API key."""

import logging

from src.config import settings

logger = logging.getLogger(__name__)

PERSONA_PROMPTS: dict[str, str] = {
    "historian": (
        "You are a knowledgeable historian and tour guide. "
        "Speak with authority, weave in dates, political context, and cultural significance. "
        "Be engaging but accurate."
    ),
    "storyteller": (
        "You are a dramatic storyteller. "
        "Bring the landmark to life with vivid imagery, myths, legends, and human drama. "
        "Make the listener feel they are part of the story."
    ),
    "local": (
        "You are a friendly Berlin local who knows the city intimately. "
        "Mix historical facts with neighbourhood vibes, hidden gems, and personal anecdotes. "
        "Be warm and conversational."
    ),
    "child_friendly": (
        "You are an enthusiastic guide speaking to children aged 8-12. "
        "Use simple language, fun facts, surprising comparisons, and a sense of wonder. "
        "Keep it short and exciting."
    ),
}


class NarrationEngine:
    """Generates tour guide narrations using Claude + RAG when available.

    Falls back to template-based narration if ANTHROPIC_API_KEY is not set,
    ensuring the pipeline always works even without API credentials.
    """

    def generate(
        self,
        scene_result: dict,
        landmark: dict | None,
        coordinates: tuple[float, float],
        max_length: int = 200,
        persona: str = "historian",
        context_chunks: list[str] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Generate a narration string for the given scene and location.

        Args:
            scene_result: Output from SceneClassifier with 'primary' key.
            landmark: Nearest landmark dict from ContextRetriever, or None.
            coordinates: (latitude, longitude) of current position.
            max_length: Soft cap on narration length in characters.
            persona: One of historian / storyteller / local / child_friendly.
            context_chunks: RAG-retrieved text chunks for grounding.
            conversation_history: Prior turns for conversational context.

        Returns:
            Narration string.
        """
        if settings.anthropic_api_key:
            try:
                return self._llm_narration(
                    scene_result, landmark, coordinates,
                    max_length, persona, context_chunks or [],
                    conversation_history or [],
                )
            except Exception as exc:
                logger.warning("Claude narration failed, falling back to template: %s", exc)

        return self._template_narration(scene_result, landmark, coordinates, max_length)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _llm_narration(
        self,
        scene_result: dict,
        landmark: dict | None,
        coordinates: tuple[float, float],
        max_length: int,
        persona: str,
        context_chunks: list[str],
        conversation_history: list[dict],
    ) -> str:
        """Generate narration using Claude via LangChain."""
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        scene_type = scene_result["primary"]["category"]
        confidence = scene_result["primary"]["confidence"]
        lat, lng = coordinates

        system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["historian"])
        system_prompt += (
            f"\n\nKeep your response under {max_length} characters. "
            "Be vivid but concise."
        )

        context_block = ""
        if context_chunks:
            joined = "\n\n".join(f"- {c}" for c in context_chunks)
            context_block = f"\n\nRelevant context from knowledge base:\n{joined}"

        landmark_block = ""
        if landmark:
            landmark_block = (
                f"\n\nLandmark: {landmark['name']}"
                f"\nDistrict: {landmark.get('district', 'unknown')}"
                f"\nCategory: {landmark.get('category', 'unknown')}"
                f"\nHistorical period: {landmark.get('historical_period', 'unknown')}"
                f"\nDistance: {landmark.get('distance_meters', 0):.0f}m from user"
            )
        else:
            landmark_block = "\n\nNo known landmark in the database for this location."

        user_message = (
            f"The user is at coordinates ({lat:.4f}, {lng:.4f}).\n"
            f"CLIP vision model identified the scene as: {scene_type} "
            f"(confidence: {confidence:.0%})."
            f"{landmark_block}"
            f"{context_block}\n\n"
            "Generate a tour guide narration for this location."
        )

        messages = [SystemMessage(content=system_prompt)]
        for turn in conversation_history[-4:]:  # last 4 turns for context
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=user_message))

        llm = ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key,
            max_tokens=400,
        )
        response = llm.invoke(messages)
        return response.content  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Template fallback path (no API key needed)
    # ------------------------------------------------------------------

    def _template_narration(
        self,
        scene_result: dict,
        landmark: dict | None,
        coordinates: tuple[float, float],
        max_length: int,
    ) -> str:
        """Build narration from DB templates — no API key required."""
        scene_type = scene_result["primary"]["category"]
        confidence = scene_result["primary"]["confidence"]

        if landmark:
            parts: list[str] = []
            if confidence >= 0.5:
                parts.append(f"I can see you're looking at a {scene_type}.")
            else:
                parts.append(f"This appears to be a {scene_type}.")
            template = landmark.get("narration_template", "")
            if template:
                parts.append(template)
            period = landmark.get("historical_period", "")
            if period:
                parts.append(f"This site dates from the {period}.")
            narration = " ".join(parts)
        else:
            lat, lng = coordinates
            lat_dir = "N" if lat >= 0 else "S"
            lng_dir = "E" if lng >= 0 else "W"
            narration = (
                f"You're exploring a {scene_type} at "
                f"{abs(lat):.4f}°{lat_dir}, {abs(lng):.4f}°{lng_dir}. "
                "This location isn't in our landmarks database yet, "
                "but keep exploring — every great city has hidden gems waiting to be discovered."
            )

        if max_length and len(narration) > max_length:
            narration = narration[:max_length].rsplit(" ", 1)[0] + "…"

        return narration
