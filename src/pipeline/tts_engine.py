"""Text-to-speech engine — OpenAI TTS (primary) with gTTS fallback."""

import io
import logging
import re

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "it", "nl", "pl", "pt"}

# OpenAI TTS voice — "alloy" is neutral and clear
_OPENAI_VOICE = "alloy"
_OPENAI_TTS_MODEL = "tts-1"


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting so TTS reads clean prose."""
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`[^`]*`", "", text)
    # Remove links, keep label
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    # Remove bullet/numbered list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove "Source: ..." lines from web search results
    text = re.sub(r"^Source:.*$", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TTSEngine:
    """Converts narration strings to MP3 audio bytes.

    Uses OpenAI TTS when OPENAI_API_KEY is configured (recommended).
    Falls back to gTTS (Google TTS, free, no key needed) otherwise.
    Strips markdown formatting from text before synthesis.
    """

    def synthesize(self, text: str, language: str = "en") -> bytes:
        """Convert text to MP3 audio bytes.

        Args:
            text: Narration string to synthesize (markdown is stripped automatically).
            language: BCP-47 language code (default: en).

        Returns:
            MP3 audio as bytes.

        Raises:
            RuntimeError: If synthesis fails via both backends.
        """
        from src.config import settings

        clean_text = _strip_markdown(text)
        if not clean_text:
            raise RuntimeError("No speakable text after stripping markdown.")

        if settings.openai_api_key:
            return self._openai_tts(clean_text)
        return self._gtts_fallback(clean_text, language)

    def _openai_tts(self, text: str) -> bytes:
        """Synthesize via OpenAI TTS API (tts-1 model)."""
        from openai import OpenAI

        from src.config import settings

        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.audio.speech.create(
                model=_OPENAI_TTS_MODEL,
                voice=_OPENAI_VOICE,
                input=text,
            )
            audio_bytes = response.content
            logger.info(
                "OpenAI TTS: %d chars → %d bytes MP3", len(text), len(audio_bytes)
            )
            return audio_bytes
        except Exception as exc:
            logger.error("OpenAI TTS failed: %s", exc)
            raise RuntimeError(f"OpenAI TTS failed: {exc}") from exc

    def _gtts_fallback(self, text: str, language: str) -> bytes:
        """Synthesize via gTTS (Google TTS, free fallback)."""
        from gtts import gTTS

        lang = language if language in SUPPORTED_LANGUAGES else "en"
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_bytes = buf.read()
            logger.info(
                "gTTS: %d chars → %d bytes MP3", len(text), len(audio_bytes)
            )
            return audio_bytes
        except Exception as exc:
            logger.error("gTTS failed: %s", exc)
            raise RuntimeError(f"gTTS synthesis failed: {exc}") from exc
