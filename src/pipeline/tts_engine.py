"""Text-to-speech engine — converts narration text to audio using gTTS (free)."""

import io
import logging

from gtts import gTTS

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "it", "nl", "pl", "pt"}


class TTSEngine:
    """Converts narration strings to MP3 audio bytes using Google TTS.

    Free, no API key required. Requires internet access.
    Returns raw MP3 bytes suitable for streaming or saving.
    """

    def synthesize(self, text: str, language: str = "en") -> bytes:
        """Convert text to MP3 audio bytes.

        Args:
            text: Narration string to synthesize.
            language: BCP-47 language code (default: en).

        Returns:
            MP3 audio as bytes.

        Raises:
            RuntimeError: If synthesis fails.
        """
        lang = language if language in SUPPORTED_LANGUAGES else "en"
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_bytes = buf.read()
            logger.info("TTS synthesized %d chars → %d bytes MP3", len(text), len(audio_bytes))
            return audio_bytes
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
            raise RuntimeError(f"TTS synthesis failed: {exc}") from exc
