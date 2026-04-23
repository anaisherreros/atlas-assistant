from __future__ import annotations

import logging
import os
import json
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atlas_client import (
    complete_task,
    create_desire,
    create_goal,
    create_habit,
    create_phase,
    create_task,
    create_transaction,
    get_dashboard,
    get_today,
    log_habit,
    log_health,
)
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
MAX_TOOL_LOOPS = 6

ATLAS_TOOLS: list[dict[str, Any]] = [
    {
        "name": "create_desire",
        "description": "Crea un nuevo deseo en Atlas Vital",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "create_task",
        "description": "Crea una tarea en Atlas Vital",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "due_date": {"type": "string", "format": "date"},
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "description": {"type": "string"},
            },
            "required": ["title", "due_date"],
        },
    },
    {
        "name": "create_habit",
        "description": "Crea un hábito en Atlas Vital",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "frequency_type": {
                    "type": "string",
                    "enum": ["daily", "weekly", "monthly"],
                },
            },
            "required": ["title", "start_date", "frequency_type"],
        },
    },
    {
        "name": "log_health",
        "description": "Registra datos de salud de hoy",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date"},
                "physical": {
                    "type": "object",
                    "properties": {
                        "weight_kg": {"type": "number"},
                        "sleep_hours": {"type": "number"},
                        "steps": {"type": "integer"},
                        "heart_rate": {"type": "integer"},
                    },
                },
                "emotional": {
                    "type": "object",
                    "properties": {
                        "mood": {"type": "integer", "minimum": 1, "maximum": 5},
                        "energy_level": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                },
                "mental": {
                    "type": "object",
                    "properties": {
                        "stress_level": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "mental_clarity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                },
            },
            "required": ["date"],
        },
    },
    {
        "name": "create_transaction",
        "description": "Registra una transacción financiera",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "amount": {"type": "number"},
                "transaction_type": {
                    "type": "string",
                    "enum": ["income", "expense"],
                },
                "date": {"type": "string", "format": "date"},
            },
            "required": ["description", "amount", "transaction_type", "date"],
        },
    },
    {
        "name": "log_habit_completion",
        "description": "Marca un hábito como completado hoy",
        "input_schema": {
            "type": "object",
            "properties": {
                "habit_id": {"type": "integer"},
                "date": {"type": "string", "format": "date"},
                "completed": {"type": "boolean"},
                "note": {"type": "string"},
            },
            "required": ["habit_id", "date", "completed"],
        },
    },
    {
        "name": "complete_task",
        "description": "Marca una tarea como completada",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "get_today",
        "description": "Obtiene tareas y hábitos de hoy",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_goal",
        "description": "Crea un objetivo dentro de una fase",
        "input_schema": {
            "type": "object",
            "properties": {
                "phase_id": {"type": "integer"},
                "title": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
            },
            "required": ["phase_id", "title", "start_date", "end_date"],
        },
    },
    {
        "name": "create_phase",
        "description": "Crea una fase dentro de un deseo",
        "input_schema": {
            "type": "object",
            "properties": {
                "desire_id": {"type": "integer"},
                "title": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
            },
            "required": ["desire_id", "title", "start_date", "end_date"],
        },
    },
]


def build_system_prompt(dashboard_data: str) -> str:
    return (
        "Eres el asistente personal de Anaïs.\n"
        "Eres su mano derecha, directa y práctica.\n\n"
        "CONTEXTO ACTUAL DE SU VIDA (datos reales de Atlas Vital):\n"
        f"{dashboard_data}\n\n"
        "Usa estos datos para dar respuestas personalizadas.\n"
        "Si los datos están vacíos en algún área, simplemente no los menciones.\n\n"
        "ACCIONES QUE PUEDES REALIZAR EN ATLAS VITAL (herramientas con tool use):\n"
        "- create_desire, create_task, create_habit\n"
        "- log_health, create_transaction, log_habit_completion, complete_task\n"
        "- get_today, create_goal, create_phase\n\n"
        "Cuando el usuario te pida crear, registrar, completar o consultar algo en Atlas,\n"
        "usa la herramienta correcta, ejecuta y confirma con precisión qué se guardó.\n\n"
        "Si la consulta del usuario trata sobre hoy, qué tiene o su día,\n"
        "prioriza usar get_today() en lugar de get_dashboard() completo."
    )


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def is_today_query(text: str) -> bool:
    normalized = text.lower()
    keywords = ("hoy", "qué tengo", "que tengo", "mi día", "mi dia")
    return any(keyword in normalized for keyword in keywords)


def _serialize_assistant_content(content: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return blocks


async def _run_atlas_tool(name: str, tool_input: dict[str, Any]) -> Any:
    if name == "create_desire":
        return await create_desire(
            title=tool_input["title"],
            description=tool_input.get("description", ""),
        )
    if name == "create_task":
        return await create_task(
            title=tool_input["title"],
            due_date=tool_input["due_date"],
            description=tool_input.get("description", ""),
            priority=tool_input.get("priority", "medium"),
        )
    if name == "create_habit":
        return await create_habit(
            title=tool_input["title"],
            start_date=tool_input["start_date"],
            frequency_type=tool_input["frequency_type"],
        )
    if name == "log_habit_completion":
        return await log_habit(
            habit_id=tool_input["habit_id"],
            date=tool_input["date"],
            completed=tool_input["completed"],
            note=tool_input.get("note", ""),
        )
    if name == "log_health":
        return await log_health(
            date=tool_input["date"],
            physical=tool_input.get("physical"),
            emotional=tool_input.get("emotional"),
            mental=tool_input.get("mental"),
        )
    if name == "create_transaction":
        return await create_transaction(
            description=tool_input["description"],
            amount=tool_input["amount"],
            transaction_type=tool_input["transaction_type"],
            date=tool_input["date"],
        )
    if name == "complete_task":
        return await complete_task(task_id=tool_input["task_id"])
    if name == "get_today":
        return await get_today()
    if name == "create_goal":
        return await create_goal(
            phase_id=tool_input["phase_id"],
            title=tool_input["title"],
            start_date=tool_input["start_date"],
            end_date=tool_input["end_date"],
        )
    if name == "create_phase":
        return await create_phase(
            desire_id=tool_input["desire_id"],
            title=tool_input["title"],
            start_date=tool_input["start_date"],
            end_date=tool_input["end_date"],
        )
    raise ValueError(f"Herramienta no soportada: {name}")


async def generate_with_tools(
    client: AsyncAnthropic,
    *,
    system_prompt: str,
    api_messages: list[dict[str, Any]],
) -> str:
    conversation_messages: list[dict[str, Any]] = list(api_messages)

    for _ in range(MAX_TOOL_LOOPS):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=system_prompt,
            tools=ATLAS_TOOLS,
            messages=conversation_messages,
        )
        assistant_text_parts: list[str] = []
        assistant_content = _serialize_assistant_content(list(response.content))
        tool_result_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                assistant_text_parts.append(block.text)
                continue
            if block.type != "tool_use":
                continue

            try:
                result = await _run_atlas_tool(block.name, block.input)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            except Exception as exc:
                logger.exception("Error ejecutando tool de Atlas Vital: %s", block.name)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error al ejecutar {block.name}: {exc}",
                        "is_error": True,
                    }
                )

        conversation_messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
            }
        )

        if tool_result_blocks:
            conversation_messages.append(
                {
                    "role": "user",
                    "content": tool_result_blocks,
                }
            )
            continue

        assistant_text = "".join(assistant_text_parts).strip()
        if assistant_text:
            return assistant_text
        return "(Sin contenido de texto en la respuesta.)"

    return "No pude completar la accion solicitada tras varios intentos de herramientas."


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
            if is_today_query(text):
                dashboard = await get_today()
            else:
                dashboard = await get_dashboard()
            dashboard_data = json.dumps(dashboard, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Error al consultar Atlas Vital")

        try:
            assistant_text = await generate_with_tools(
                client,
                system_prompt=build_system_prompt(dashboard_data),
                api_messages=api_messages,
            )
        except Exception:
            logger.exception("Error al llamar a la API de Anthropic")
            await update.message.reply_text(
                "No pude obtener respuesta del asistente ahora mismo. "
                "Inténtalo de nuevo en unos segundos."
            )
            return

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
