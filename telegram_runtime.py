from __future__ import annotations

import asyncio
import contextlib
import logging
import os

from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes

from conversation_flow import UserFacingError, process_text_message
from database import create_engine, init_db, session_factory
from daily_automation import start_daily_automation, stop_daily_automation

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TYPING_RENEWAL_SECONDS = float(os.getenv("TYPING_RENEWAL_SECONDS", "4.5"))
PROVISIONAL_REPLY_DELAY_SECONDS = float(os.getenv("PROVISIONAL_REPLY_DELAY_SECONDS", "8"))
PROVISIONAL_REPLY_TEXT = os.getenv("PROVISIONAL_REPLY_TEXT", "Dame un segundo…")


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _typing_action_loop(bot, chat_id: int, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            logger.exception("No se pudo enviar chat action TYPING (chat %s)", chat_id)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=TYPING_RENEWAL_SECONDS)
            return
        except asyncio.TimeoutError:
            continue


async def _send_provisional_message(
    message: Message,
    *,
    delay_seconds: float,
    stop_event: asyncio.Event,
) -> Message | None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=delay_seconds)
        return None
    except asyncio.TimeoutError:
        pass
    if stop_event.is_set():
        return None
    try:
        return await message.reply_text(PROVISIONAL_REPLY_TEXT)
    except Exception:
        logger.exception("No se pudo enviar mensaje provisional")
        return None


async def _deliver_reply(
    update: Update,
    *,
    reply_messages: list[str],
    provisional_message: Message | None,
) -> None:
    if update.message is None:
        return

    parts: list[str] = []
    for message in reply_messages:
        parts.extend(chunk_text(message, TELEGRAM_MAX_MESSAGE_LENGTH))
    if not parts:
        return

    if provisional_message is not None:
        await provisional_message.edit_text(parts[0])
        for part in parts[1:]:
            await update.message.reply_text(part)
        return

    for part in parts:
        await update.message.reply_text(part)


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

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(
        _typing_action_loop(context.bot, chat_id, stop_event),
        name=f"typing-{chat_id}",
    )
    provisional_task = asyncio.create_task(
        _send_provisional_message(
            update.message,
            delay_seconds=PROVISIONAL_REPLY_DELAY_SECONDS,
            stop_event=stop_event,
        ),
        name=f"provisional-{chat_id}",
    )

    result = None
    try:
        async with session_factory_() as session:
            result = await process_text_message(
                session,
                client=client,
                text=text,
                chat_id=chat_id,
                user_id=user_id,
                session_factory=session_factory_,
            )
    except UserFacingError as exc:
        stop_event.set()
        typing_task.cancel()
        provisional_message = await provisional_task
        if provisional_message is not None:
            await provisional_message.edit_text(str(exc))
        else:
            await update.message.reply_text(str(exc))
        return
    finally:
        stop_event.set()
        typing_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await typing_task

    provisional_message = await provisional_task

    if result is None:
        return

    if result.model_used:
        logger.info(
            "Respuesta entregada chat=%s model=%s tier=%s",
            chat_id,
            result.model_used,
            result.model_tier_reason,
        )

    await _deliver_reply(
        update,
        reply_messages=result.reply_messages,
        provisional_message=provisional_message,
    )
