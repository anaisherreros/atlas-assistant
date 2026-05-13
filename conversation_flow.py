from __future__ import annotations

import logging
from dataclasses import dataclass

from agent_tool_policy import get_tools_for_agent
from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession

from agents import build_agent_system_prompt, fetch_context_for_agent, get_agent
from claude import generate_with_tools
from database import (
    ensure_chat_session,
    fetch_conversation_messages,
    get_active_agent,
    messages_to_anthropic,
    save_message,
    set_active_agent,
)
from deterministic_handlers import try_handle_deterministic_message
from message_classification import classify_context, classify_message
from router import TRANSITION_MESSAGES, detect_agent

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"
MAX_HISTORY_MESSAGES = 20
MAX_TOOL_LOOPS = 12


class UserFacingError(Exception):
    """Error con mensaje listo para mostrar al usuario."""


@dataclass(frozen=True)
class ConversationResult:
    reply_messages: list[str]


def _history_limit_for_context(context_kind: str) -> int:
    if context_kind == "none":
        return 0
    if context_kind == "today":
        return 5
    if context_kind == "finance":
        return 8
    return MAX_HISTORY_MESSAGES


def _should_persist_response(assistant_text: str, tools_used: bool) -> bool:
    response_words = len(assistant_text.split())
    is_action = tools_used and response_words < 100
    return not is_action


async def _build_api_messages(
    session: AsyncSession,
    *,
    chat_id: int,
    text: str,
    context_kind: str,
    history_limit: int,
) -> list[dict[str, str]]:
    history_rows = await fetch_conversation_messages(
        session,
        telegram_chat_id=chat_id,
        limit=history_limit,
    )
    api_messages = messages_to_anthropic(history_rows)
    if context_kind == "none":
        return [{"role": "user", "content": text}]
    api_messages.append({"role": "user", "content": text})
    return api_messages


async def process_text_message(
    session: AsyncSession,
    *,
    client: AsyncAnthropic,
    text: str,
    chat_id: int,
    user_id: int,
) -> ConversationResult:
    outgoing_messages: list[str] = []

    previous_agent = await get_active_agent(session, telegram_chat_id=chat_id)
    selected_agent = detect_agent(text, previous_agent)
    await ensure_chat_session(
        session,
        telegram_chat_id=chat_id,
        active_agent=selected_agent,
    )
    if selected_agent != previous_agent:
        transition = TRANSITION_MESSAGES.get(
            selected_agent,
            "Cambiando de agente...",
        )
        outgoing_messages.append(transition)
        await set_active_agent(
            session,
            telegram_chat_id=chat_id,
            active_agent=selected_agent,
        )
        logger.info("Agente activo: %s → %s", previous_agent, selected_agent)

    try:
        deterministic_result = await try_handle_deterministic_message(text)
    except Exception as exc:
        logger.exception("Error en la ruta determinista")
        raise UserFacingError(
            "No pude completar esa consulta o acción de Atlas ahora mismo. "
            "Inténtalo de nuevo en unos segundos."
        ) from exc

    if deterministic_result is not None:
        logger.info("Mensaje resuelto por ruta determinista.")
        if deterministic_result.persist_conversation:
            await save_message(
                session,
                telegram_chat_id=chat_id,
                telegram_user_id=user_id,
                role="user",
                content=text,
            )
            await save_message(
                session,
                telegram_chat_id=chat_id,
                telegram_user_id=user_id,
                role="assistant",
                content=deterministic_result.text,
            )
        outgoing_messages.append(deterministic_result.text)
        return ConversationResult(reply_messages=outgoing_messages)

    agent = get_agent(selected_agent)
    context_kind = classify_context(text)
    history_limit = _history_limit_for_context(context_kind)
    api_messages = await _build_api_messages(
        session,
        chat_id=chat_id,
        text=text,
        context_kind=context_kind,
        history_limit=history_limit,
    )

    logger.info(
        "Contexto Atlas (precarga classify_context): %s | agente: %s",
        context_kind,
        selected_agent,
    )
    logger.info("Historial chat: limit=%s (mensajes=%d)", history_limit, len(api_messages))

    context_data = await fetch_context_for_agent(selected_agent)
    system_prompt = build_agent_system_prompt(agent, context_data)

    try:
        complexity = classify_message(text)
        agent_tools = get_tools_for_agent(selected_agent)
        logger.info("Clasificacion de mensaje: %s (modelo: %s)", complexity, MODEL)
        logger.info("Modelo elegido: %s para: %s", MODEL, text[:50])
        assistant_text, tools_used = await generate_with_tools(
            client,
            model=MODEL,
            system_prompt=system_prompt,
            api_messages=api_messages,
            tools=agent_tools,
            max_tool_loops=MAX_TOOL_LOOPS,
        )
    except Exception as exc:
        logger.exception("Error al llamar a la API de Anthropic")
        raise UserFacingError(
            "No pude obtener respuesta del asistente ahora mismo. "
            "Inténtalo de nuevo en unos segundos."
        ) from exc

    if _should_persist_response(assistant_text, tools_used):
        await save_message(
            session,
            telegram_chat_id=chat_id,
            telegram_user_id=user_id,
            role="user",
            content=text,
        )
        await save_message(
            session,
            telegram_chat_id=chat_id,
            telegram_user_id=user_id,
            role="assistant",
            content=assistant_text,
        )

    outgoing_messages.append(assistant_text)
    return ConversationResult(reply_messages=outgoing_messages)
