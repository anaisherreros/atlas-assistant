from __future__ import annotations

import io
import logging
import os
import time

from openai import APITimeoutError, AsyncOpenAI

logger = logging.getLogger(__name__)

WHISPER_TIMEOUT_SECONDS = 30.0


class VoiceTranscriptionError(Exception):
    """Error controlado al transcribir un mensaje de voz."""


def is_voice_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def get_whisper_model() -> str:
    return os.getenv("WHISPER_MODEL", "whisper-1").strip() or "whisper-1"


async def transcribe_voice_message(file_bytes: bytes, filename: str = "audio.ogg") -> str:
    """
    Transcribe un audio de Telegram usando Whisper.
    Lanza VoiceTranscriptionError si falla.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise VoiceTranscriptionError("Los mensajes de voz no están configurados aún")

    if not file_bytes:
        raise VoiceTranscriptionError("No detecté voz en el audio, ¿puedes repetirlo?")

    size_kb = len(file_bytes) / 1024
    logger.info("Whisper: iniciando transcripción %s (%.1f KB)", filename, size_kb)

    client = AsyncOpenAI(api_key=api_key, timeout=WHISPER_TIMEOUT_SECONDS)
    buffer = io.BytesIO(file_bytes)
    buffer.name = filename
    started = time.monotonic()

    try:
        response = await client.audio.transcriptions.create(
            model=get_whisper_model(),
            file=buffer,
        )
    except APITimeoutError as exc:
        elapsed = time.monotonic() - started
        logger.warning("Whisper: timeout tras %.1fs (%s)", elapsed, filename)
        raise VoiceTranscriptionError("No pude entender el audio, ¿puedes escribirlo?") from exc
    except Exception as exc:
        elapsed = time.monotonic() - started
        logger.exception("Whisper: error tras %.1fs (%s)", elapsed, filename)
        raise VoiceTranscriptionError("No pude entender el audio, ¿puedes escribirlo?") from exc

    elapsed = time.monotonic() - started
    text = (response.text or "").strip()
    preview = text[:100] + ("…" if len(text) > 100 else "")
    logger.info("Whisper: transcripción lista en %.1fs — %r", elapsed, preview)

    if not text:
        raise VoiceTranscriptionError("No detecté voz en el audio, ¿puedes repetirlo?")

    return text
