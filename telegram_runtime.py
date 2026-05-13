from __future__ import annotations

import logging
import os

from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes

from conversation_flow import UserFacingError, process_text_message
from database import create_engine, init_db, session_factory
from daily_automation import start_daily_automation, stop_daily_automation

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


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
    start_daily_automation(application)
    logger.info("Base de datos lista y cliente Anthropic configurado.")


async def post_shutdown(application: Application) -> None:
    await stop_daily_automation(application)
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
        try:
            result = await process_text_message(
                session,
                client=client,
                text=text,
                chat_id=chat_id,
                user_id=user_id,
            )
        except UserFacingError as exc:
            await update.message.reply_text(str(exc))
            return

    for message in result.reply_messages:
        for part in chunk_text(message, TELEGRAM_MAX_MESSAGE_LENGTH):
            await update.message.reply_text(part)
