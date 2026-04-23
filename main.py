from __future__ import annotations

import logging
import os
import json

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atlas_client import get_dashboard
from database import (
    create_engine,
    fetch_conversation_messages,
    init_db,
    messages_to_anthropic,
    save_message,
    session_factory,
)

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"
MAX_HISTORY_MESSAGES = 60
TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def build_system_prompt(dashboard_data: str) -> str:
    return (
        "Eres el asistente personal de Anaïs.\n"
        "Eres su mano derecha, directa y práctica.\n\n"
        "CONTEXTO ACTUAL DE SU VIDA (datos reales de Atlas Vital):\n"
        f"{dashboard_data}\n\n"
        "Usa estos datos para dar respuestas personalizadas.\n"
        "Si los datos están vacíos en algún área, simplemente no los menciones."
    )


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


async def post_init(application: Application) -> None:
    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    await init_db(engine)
    application.bot_data["engine"] = engine
    application.bot_data["session_factory"] = session_factory(engine)
    application.bot_data["anthropic"] = AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    logger.info("Base de datos lista y cliente Anthropic configurado.")


async def post_shutdown(application: Application) -> None:
    engine = application.bot_data.get("engine")
    if engine is not None:
        await engine.dispose()
    logger.info("Motor SQLAlchemy cerrado.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_chat is None:
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else chat_id

    session_factory_: async_sessionmaker[AsyncSession] = context.application.bot_data[
        "session_factory"
    ]
    client: AsyncAnthropic = context.application.bot_data["anthropic"]

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    async with session_factory_() as session:
        await save_message(
            session,
            telegram_chat_id=chat_id,
            telegram_user_id=user_id,
            role="user",
            content=text,
        )

        history_rows = await fetch_conversation_messages(
            session,
            telegram_chat_id=chat_id,
            limit=MAX_HISTORY_MESSAGES,
        )
        api_messages = messages_to_anthropic(history_rows)
        dashboard_data = "{}"

        try:
            dashboard = await get_dashboard()
            dashboard_data = json.dumps(dashboard, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Error al consultar Atlas Vital")

        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=8192,
                system=build_system_prompt(dashboard_data),
                messages=api_messages,
            )
        except Exception:
            logger.exception("Error al llamar a la API de Anthropic")
            await update.message.reply_text(
                "No pude obtener respuesta del asistente ahora mismo. "
                "Inténtalo de nuevo en unos segundos."
            )
            return

        assistant_text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                assistant_text_parts.append(block.text)
        assistant_text = "".join(assistant_text_parts).strip()
        if not assistant_text:
            assistant_text = "(Sin contenido de texto en la respuesta.)"

        await save_message(
            session,
            telegram_chat_id=chat_id,
            telegram_user_id=user_id,
            role="assistant",
            content=assistant_text,
        )

    for part in chunk_text(assistant_text, TELEGRAM_MAX_MESSAGE_LENGTH):
        await update.message.reply_text(part)


def main() -> None:
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

    token = os.environ["TELEGRAM_BOT_TOKEN"]

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
