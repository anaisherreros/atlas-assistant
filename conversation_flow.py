from __future__ import annotations

import logging
from dataclasses import dataclass

from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import AsyncSession

from agents import build_agent_system_prompt, get_agent
from claude import generate_with_tools
from database import (
    MEMORY_UPDATE_EVERY_USER_MESSAGES,
    count_user_messages,
    ensure_chat_session,
    fetch_conversation_messages,
    get_memory,
    get_active_agent,
    messages_to_anthropic,
    save_message,
    save_memory,
    should_update_memory,
    set_active_agent,
    update_memory,
)
from deterministic_handlers import try_handle_deterministic_message
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


def _should_persist_response(assistant_text: str, tools_used: bool) -> bool:
    return not (tools_used and len(assistant_text.split()) < 100)


def _memory_context_block(summary: str) -> str:
    return (
        "MEMORIA DE CONVERSACIONES ANTERIORES:\n"
        f"{summary}\n\n"
        "Usa esto para personalizar respuestas "
        "y detectar patrones a lo largo del tiempo."
    )


async def _generate_memory_summary(client: AsyncAnthropic, *, conversation_text: str) -> str:
    prompt = (
        "Resume en máximo 300 palabras lo más importante de esta conversación:\n"
        "patrones detectados, decisiones tomadas, estado emocional, avances, bloqueos,\n"
        "temas recurrentes. Solo lo que sea útil recordar en el futuro.\n\n"
        "Captura:\n"
        "- Patrones emocionales detectados\n"
        "- Decisiones importantes tomadas\n"
        "- Avances en objetivos\n"
        "- Bloqueos recurrentes\n"
        "- Estado general de las áreas de vida\n"
        "- Temas que se repiten\n\n"
        f"Conversación:\n{conversation_text}"
    )
    response = await client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system="Eres un analista de memoria conversacional. Responde solo con el resumen.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


async def _merge_memory_summaries(client: AsyncAnthropic, *, old_summary: str, new_summary: str) -> str:
    prompt = (
        f"Resumen anterior: {old_summary}\n\n"
        f"Nuevos puntos importantes: {new_summary}\n\n"
        "Genera un resumen actualizado y completo de máximo 400 palabras."
    )
    response = await client.messages.create(
        model=MODEL,
        max_tokens=1400,
        system="Fusiona memorias de conversación en una versión acumulativa, clara y útil.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _extract_key_facts_json(summary: str) -> str:
    escaped = summary.replace("\\", "\\\\").replace('"', '\\"')
    return f'{{"always_remember":"{escaped}"}}'


async def _update_conversation_memory_if_needed(
    session: AsyncSession,
    *,
    client: AsyncAnthropic,
    chat_id: int,
) -> None:
    user_message_count = await count_user_messages(session, telegram_chat_id=chat_id)
    if not should_update_memory(user_message_count):
        return

    logger.info("Memoria: generando resumen tras %d mensajes (chat %s)", user_message_count, chat_id)
    try:
        history_rows = await fetch_conversation_messages(
            session,
            telegram_chat_id=chat_id,
            limit=MEMORY_UPDATE_EVERY_USER_MESSAGES * 2,
        )
        if not history_rows:
            return

        conversation_text = "\n".join(
            f"{row.role}: {row.content}" for row in history_rows if row.content
        ).strip()
        if not conversation_text:
            return

        new_summary = await _generate_memory_summary(client, conversation_text=conversation_text)
        if not new_summary:
            return

        existing_memory = await get_memory(session, telegram_chat_id=chat_id)
        if existing_memory is None or not existing_memory.summary:
            await save_memory(
                session,
                telegram_chat_id=chat_id,
                summary=new_summary,
                key_facts=_extract_key_facts_json(new_summary),
                user_message_count=user_message_count,
            )
            return

        merged = await _merge_memory_summaries(client, old_summary=existing_memory.summary, new_summary=new_summary)
        if merged:
            await update_memory(
                session,
                telegram_chat_id=chat_id,
                new_info=merged,
                key_facts=_extract_key_facts_json(merged),
                user_message_count=user_message_count,
            )
    except Exception:
        logger.exception("Memoria: error al generar o guardar resumen (chat %s)", chat_id)


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
    await ensure_chat_session(session, telegram_chat_id=chat_id, active_agent=selected_agent)

    if selected_agent != previous_agent:
        transition = TRANSITION_MESSAGES.get(selected_agent, "Cambiando de agente...")
        outgoing_messages.append(transition)
        await set_active_agent(session, telegram_chat_id=chat_id, active_agent=selected_agent)
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
            await save_message(session, telegram_chat_id=chat_id, telegram_user_id=user_id, role="user", content=text)
            await save_message(session, telegram_chat_id=chat_id, telegram_user_id=user_id, role="assistant", content=deterministic_result.text)
            await _update_conversation_memory_if_needed(session, client=client, chat_id=chat_id)
        outgoing_messages.append(deterministic_result.text)
        return ConversationResult(reply_messages=outgoing_messages)

    agent = get_agent(selected_agent)
    history_rows = await fetch_conversation_messages(session, telegram_chat_id=chat_id, limit=MAX_HISTORY_MESSAGES)
    api_messages = messages_to_anthropic(history_rows)
    api_messages.append({"role": "user", "content": text})

    system_prompt = build_agent_system_prompt(agent)
    memory = await get_memory(session, telegram_chat_id=chat_id)
    if memory is not None and memory.summary:
        system_prompt = f"{system_prompt}\n\n{_memory_context_block(memory.summary)}"

    try:
        assistant_text, tools_used = await generate_with_tools(
            client,
            model=MODEL,
            system_prompt=system_prompt,
            api_messages=api_messages,
            max_tool_loops=MAX_TOOL_LOOPS,
        )
    except Exception as exc:
        logger.exception("Error al llamar a la API de Anthropic")
        raise UserFacingError(
            "No pude obtener respuesta del asistente ahora mismo. "
            "Inténtalo de nuevo en unos segundos."
        ) from exc

    if _should_persist_response(assistant_text, tools_used):
        await save_message(session, telegram_chat_id=chat_id, telegram_user_id=user_id, role="user", content=text)
        await save_message(session, telegram_chat_id=chat_id, telegram_user_id=user_id, role="assistant", content=assistant_text)
        await _update_conversation_memory_if_needed(session, client=client, chat_id=chat_id)

    outgoing_messages.append(assistant_text)
    return ConversationResult(reply_messages=outgoing_messages)
