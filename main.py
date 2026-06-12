from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from telegram_runtime import handle_text, handle_user_message, post_init, post_shutdown
from voice import VoiceTranscriptionError, is_voice_configured, transcribe_voice_message

load_dotenv()

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

VOICE_MAX_DURATION_SECONDS = 300
VOICE_NOT_CONFIGURED_MESSAGE = "Los mensajes de voz no están configurados aún"
VOICE_TOO_LONG_MESSAGE = "El audio es demasiado largo (máx. 5 min)"
VOICE_TRANSCRIPTION_VISIBLE_CHARS = 200

_MIME_TO_EXTENSION = {
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
}


def validate_environment() -> None:
    required = (
        "TELEGRAM_BOT_TOKEN",
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "ATLAS_VITAL_URL",
        "ASSISTANT_API_KEY",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Faltan variables de entorno obligatorias: " + ", ".join(missing)
        )


def check_voice_configuration() -> None:
    if not is_voice_configured():
        logger.warning("OPENAI_API_KEY no configurada — mensajes de voz desactivados")


def _voice_show_transcription_enabled() -> bool:
    return os.getenv("VOICE_SHOW_TRANSCRIPTION", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _escape_markdown_for_italic(text: str) -> str:
    return text.replace("\\", "\\\\").replace("_", "\\_").replace("*", "\\*")


def _audio_filename_from_message(*, voice, audio) -> tuple[str, str]:
    if voice is not None:
        return voice.file_id, "audio.ogg"

    assert audio is not None
    file_name = (audio.file_name or "").strip()
    if file_name:
        return audio.file_id, file_name

    mime_type = (getattr(audio, "mime_type", None) or "").strip().lower()
    extension = _MIME_TO_EXTENSION.get(mime_type, ".m4a")
    return audio.file_id, f"audio{extension}"


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_chat is None:
        return

    if not is_voice_configured():
        await update.message.reply_text(VOICE_NOT_CONFIGURED_MESSAGE)
        return

    voice = update.message.voice
    audio = update.message.audio
    if voice is None and audio is None:
        return

    media = voice or audio
    duration = getattr(media, "duration", None) or 0
    if duration > VOICE_MAX_DURATION_SECONDS:
        await update.message.reply_text(VOICE_TOO_LONG_MESSAGE)
        return

    chat_id = update.effective_chat.id
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except Exception:
        logger.exception("No se pudo enviar chat action TYPING inicial (chat %s)", chat_id)

    file_id, filename = _audio_filename_from_message(voice=voice, audio=audio)

    try:
        tg_file = await context.bot.get_file(file_id)
        audio_bytes = await tg_file.download_as_bytearray()
        transcription = await transcribe_voice_message(bytes(audio_bytes), filename)
    except VoiceTranscriptionError as exc:
        await update.message.reply_text(str(exc))
        return
    except Exception:
        logger.exception("Error al descargar o transcribir audio (chat %s)", chat_id)
        await update.message.reply_text("No pude entender el audio, ¿puedes escribirlo?")
        return

    if _voice_show_transcription_enabled():
        visible = transcription
        if len(visible) > VOICE_TRANSCRIPTION_VISIBLE_CHARS:
            visible = visible[: VOICE_TRANSCRIPTION_VISIBLE_CHARS - 1].rstrip() + "…"
        escaped = _escape_markdown_for_italic(visible)
        try:
            await update.message.reply_text(
                f'🎙️ _"{escaped}"_',
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("No se pudo enviar transcripción con Markdown; enviando texto plano")
            await update.message.reply_text(f'🎙️ "{visible}"')

    await handle_user_message(update, context, transcription)


def build_application() -> Application:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_message)
    )
    return application


def main() -> None:
    validate_environment()
    check_voice_configuration()
    application = build_application()
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
