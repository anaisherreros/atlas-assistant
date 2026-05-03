from __future__ import annotations

import logging
import os
import json
import re
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
    get_all_desires_full,
    get_areas_full,
    get_calendar,
    get_dashboard,
    get_desire_structure,
    get_finance,
    get_finance_full,
    get_relationships_full,
    get_reviews_summary,
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

SONNET_MODEL = "claude-sonnet-4-5"
HAIKU_MODEL = "claude-haiku-4-5"
MAX_HISTORY_MESSAGES = 20
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
    {
        "name": "get_desire_structure",
        "description": (
            "Obtiene estructura completa de un deseo con sus fases, "
            "objetivos y hábitos"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "desire_id": {"type": "integer"},
            },
            "required": ["desire_id"],
        },
    },
    {
        "name": "get_all_desires_full",
        "description": (
            "Obtiene todos los deseos activos con su estructura completa anidada"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_calendar",
        "description": "Ve hábitos y tareas en un rango de fechas",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_areas_full",
        "description": "Obtiene todas las áreas y subáreas con sus IDs y slugs",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_relationships_full",
        "description": (
            "Obtiene todas las relaciones personales con historial reciente"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_reviews_summary",
        "description": (
            "Obtiene resumen de últimas revisiones diaria, semanal, "
            "mensual y anual"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_finance_full",
        "description": (
            "Obtiene presupuesto anual completo con categorías y "
            "gastos reales del mes"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_finance",
        "description": (
            "Obtiene el detalle financiero del mes en curso en Atlas Vital: "
            "lista de transacciones, totales de ingresos/gastos y balance, "
            "categorías del presupuesto con lo gastado o cobrado este mes "
            "(incluye nombres como gasolina, comida, etc.) y último snapshot "
            "de patrimonio si existe. "
            "Úsala para preguntas del tipo cuánto he gastado, cuánto en gasolina, "
            "totales del mes o desglose por categoría."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
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
        "- get_today, create_goal, create_phase\n"
        "- get_desire_structure, get_all_desires_full, get_calendar\n"
        "- get_areas_full, get_relationships_full, get_reviews_summary, "
        "get_finance_full, get_finance\n\n"
        "Cuando el usuario te pida crear, registrar, completar o consultar algo en Atlas,\n"
        "usa la herramienta correcta, ejecuta y confirma con precisión qué se guardó.\n\n"
        "Si la consulta del usuario trata sobre hoy, qué tiene o su día,\n"
        "prioriza usar get_today() en lugar de get_dashboard() completo."
    )


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def classify_context(text: str) -> str:
    """Precarga de contexto Atlas en el system prompt: none / today / full."""
    normalized = text.lower().strip()
    if not normalized:
        return "none"

    full_markers = (
        "reflexionar",
        "reflexión",
        "reflexiona",
        "analiza mis",
        "analizar mis",
        "análisis de mis",
        "analisis de mis",
        "analiza mi vida",
        "analiza mi situación",
        "analiza mi situacion",
        "coaching",
        "mis metas",
        "mis objetivos",
        "planificación",
        "planeación",
        "planeacion",
        "planifica mi",
        "balance de vida",
        "panorama general",
        "todas mis áreas",
        "mi vida en general",
        "filosofía",
        "sentido de mi vida",
        "patrones en mi",
        "desarrollo personal",
        "ayúdame a pensar",
        "ayudame a pensar",
        "piensa conmigo sobre",
        "analiza mis finanzas",
        "analizar mis finanzas",
        "revisa mi situación financiera",
        "revisa mi situacion financiera",
    )
    if any(m in normalized for m in full_markers):
        return "full"

    finance_markers = (
        "gasolina",
        "combustible",
        "diesel",
        "repostaje",
        "he gastado",
        "llevo gastado",
        "llevo gasto",
        "gasté en",
        "gaste en",
        "gastado en",
        "gasto en ",
        "cuánto he gastado",
        "cuanto he gastado",
        "cuánto llevo gastado",
        "cuanto llevo gastado",
        "cuánto gasté",
        "cuanto gaste",
        "gastos del mes",
        "ingresos del mes",
        "balance del mes",
        "mis finanzas",
        "situación financiera",
        "situacion financiera",
        "transacciones del mes",
        "movimientos del mes",
        "cuánto dinero",
        "cuanto dinero",
        "desglose por categoría",
        "desglose por categoria",
        "presupuesto del mes",
    )
    if any(m in normalized for m in finance_markers):
        return "finance"
    if re.search(
        r"\b(cuánto|cuanto)\b.*\b(gastado|gastos|gasté|gaste)\b",
        normalized,
    ):
        return "finance"
    if re.search(r"\b(gasolina|combustible|diesel)\b", normalized) and re.search(
        r"\b(cuánto|cuanto|qué|que|llevo|coste|costo)\b",
        normalized,
    ):
        return "finance"

    if re.fullmatch(
        r"(hola|hey|buenas|buenos días|buenas tardes|buenas noches)(\s*[!.¡…]*)?",
        normalized,
    ):
        return "none"
    if re.fullmatch(
        r"(hola|hey)\s*,?\s*(qué|que)\s+tal\s*[!.¡?¿]*",
        normalized,
    ):
        return "none"
    thanks_only = ("gracias", "muchas gracias", "ok", "vale", "perfecto", "genial")
    if normalized in thanks_only:
        return "none"

    conceptual_starts = (
        "qué es ",
        "que es ",
        "qué son ",
        "que son ",
        "define ",
        "define qué ",
        "define que ",
    )
    if any(normalized.startswith(s) for s in conceptual_starts):
        if " mi " not in normalized and not normalized.startswith("mi "):
            return "none"

    today_markers = (
        "qué tengo hoy",
        "que tengo hoy",
        "qué tengo para hoy",
        "que tengo para hoy",
        "marcar",
        "completar",
        "check",
        "registra",
        "apunta",
        "cuánto llevo",
        "cuanto llevo",
        "mis hábitos",
        "mis habitos",
        "mis tareas",
        "crea una tarea",
        "crea tarea",
        "nueva tarea",
        "tarea para",
        "marca el hábito",
        "marca el habito",
        "hábito",
        "habito",
        "hábitos",
        "habitos",
        "para hoy",
        "mi día",
        "mi dia",
        "calendario",
        "agenda",
        "entre fechas",
        "rango de fechas",
        "estructura del deseo",
        "estructura de mi deseo",
        "estructura de un deseo",
        "todos los deseos",
        "mis deseos activos",
        "deseos activos",
        "listado de deseos",
        "mis áreas",
        "areas de vida",
        "áreas de vida",
        "subáreas",
        "subareas",
        "mis relaciones",
        "relaciones personales",
        "historial de relaciones",
        "resumen de revisiones",
        "revisiones diaria",
        "revision semanal",
        "revisión mensual",
        "revision mensual",
        "finanzas completas",
        "presupuesto anual",
        "gastos del mes",
        "gastos reales",
        "crea un deseo",
        "crea deseo",
        "nueva fase",
        "crea fase",
    )
    if any(m in normalized for m in today_markers):
        return "today"

    if len(normalized.split()) <= 14:
        if re.match(
            r"^(crea|haz|marca|completa|registra|apunta|muestra|dime)\s+",
            normalized,
        ):
            return "today"

    if re.search(r"\b(mi|mis|me)\s+", normalized):
        return "full"

    if len(normalized.split()) > 25:
        return "full"

    return "none"


def classify_message(text: str) -> str:
    normalized = text.lower().strip()
    simple_keywords = (
        "qué tengo hoy",
        "que tengo hoy",
        "marcar",
        "completar",
        "check",
        "registra",
        "apunta",
        "cuánto llevo",
        "cuanto llevo",
        "mis hábitos",
        "mis habitos",
        "mis tareas",
        # Consultas de lectura (get_desire_structure, get_all_desires_full, etc.)
        "estructura del deseo",
        "estructura de mi deseo",
        "estructura de un deseo",
        "estructura completa del deseo",
        "fases y objetivos del deseo",
        "todos los deseos",
        "mis deseos activos",
        "deseos activos",
        "deseos completos",
        "listado de deseos",
        "calendario",
        "en el calendario",
        "entre fechas",
        "rango de fechas",
        "agenda entre",
        "mis áreas",
        "areas de vida",
        "áreas de vida",
        "subáreas",
        "subareas",
        "mis relaciones",
        "relaciones personales",
        "historial de relaciones",
        "resumen de revisiones",
        "revisiones diaria",
        "revision semanal",
        "revisión mensual",
        "revision mensual",
        "finanzas completas",
        "presupuesto anual",
        "gastos del mes",
        "gastos reales",
        "gasolina",
        "combustible",
        "he gastado",
        "llevo gastado",
        "cuánto he gastado",
        "cuanto he gastado",
        "mis finanzas",
        "balance del mes",
        "ingresos del mes",
    )
    if any(keyword in normalized for keyword in simple_keywords):
        return "simple"

    words = [word for word in normalized.split() if word]
    if len(words) < 15:
        direct_starts = (
            "que ",
            "qué ",
            "cuanto ",
            "cuánto ",
            "marca ",
            "completa ",
            "registra ",
            "apunta ",
            "crea ",
            "haz ",
            "muestra ",
            "dime ",
        )
        if normalized.endswith("?") or normalized.startswith(direct_starts):
            return "simple"

    return "complex"


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
    if name == "get_desire_structure":
        return await get_desire_structure(desire_id=tool_input["desire_id"])
    if name == "get_all_desires_full":
        return await get_all_desires_full()
    if name == "get_calendar":
        return await get_calendar(
            start_date=tool_input["start_date"],
            end_date=tool_input["end_date"],
        )
    if name == "get_areas_full":
        return await get_areas_full()
    if name == "get_relationships_full":
        return await get_relationships_full()
    if name == "get_reviews_summary":
        return await get_reviews_summary()
    if name == "get_finance_full":
        return await get_finance_full()
    if name == "get_finance":
        return await get_finance()
    raise ValueError(f"Herramienta no soportada: {name}")


async def generate_with_tools(
    client: AsyncAnthropic,
    *,
    model: str,
    system_prompt: str,
    api_messages: list[dict[str, Any]],
) -> tuple[str, bool]:
    conversation_messages: list[dict[str, Any]] = list(api_messages)
    tools_were_used = False

    for _ in range(MAX_TOOL_LOOPS):
        logger.info("Llamando a Claude con %d mensajes", len(conversation_messages))
        response = await client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            tools=ATLAS_TOOLS,
            messages=conversation_messages,
        )
        logger.info(
            "Tokens usados - input: %s output: %s",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        logger.info("Stop reason: %s", response.stop_reason)
        logger.info("Bloques en respuesta: %s", [b.type for b in response.content])
        assistant_text_parts: list[str] = []
        assistant_content = _serialize_assistant_content(list(response.content))
        tool_result_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                assistant_text_parts.append(block.text)
                continue
            if block.type != "tool_use":
                continue

            tools_were_used = True

            logger.info("Ejecutando tool: %s con input: %s", block.name, block.input)

            try:
                result = await _run_atlas_tool(block.name, block.input)
                logger.info("Resultado de tool %s: %s", block.name, result)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            except Exception as exc:
                logger.error("Error en tool %s: %s", block.name, exc)
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
            return assistant_text, tools_were_used
        return "(Sin contenido de texto en la respuesta.)", tools_were_used

    return (
        "No pude completar la accion solicitada tras varios intentos de herramientas.",
        tools_were_used,
    )


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
        ctx = classify_context(text)
        if ctx == "none":
            history_limit = 0
        elif ctx == "today":
            history_limit = 5
        elif ctx == "finance":
            history_limit = 8
        else:
            history_limit = MAX_HISTORY_MESSAGES

        history_rows = await fetch_conversation_messages(
            session,
            telegram_chat_id=chat_id,
            limit=history_limit,
        )
        api_messages = messages_to_anthropic(history_rows)
        if ctx == "none":
            api_messages = [{"role": "user", "content": text}]
        else:
            api_messages.append({"role": "user", "content": text})

        dashboard_data = "{}"
        logger.info("Contexto Atlas (precarga): %s", ctx)
        logger.info("Historial Claude: limit=%s (mensajes=%d)", history_limit, len(api_messages))

        try:
            if ctx == "full":
                dashboard = await get_dashboard()
                dashboard_data = json.dumps(
                    dashboard, ensure_ascii=False, separators=(",", ":")
                )
            elif ctx == "today":
                dashboard = await get_today()
                dashboard_data = json.dumps(
                    dashboard, ensure_ascii=False, separators=(",", ":")
                )
            elif ctx == "finance":
                dashboard = await get_finance()
                dashboard_data = json.dumps(
                    dashboard, ensure_ascii=False, separators=(",", ":")
                )
        except Exception:
            logger.exception("Error al consultar Atlas Vital")

        try:
            complexity = classify_message(text)
            model = HAIKU_MODEL if complexity == "simple" else SONNET_MODEL
            logger.info("Clasificacion de mensaje: %s (modelo: %s)", complexity, model)
            logger.info("Modelo elegido: %s para: %s", model, text[:50])
            assistant_text, tools_used = await generate_with_tools(
                client,
                model=model,
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

        response_words = len(assistant_text.split())
        is_action = tools_used and response_words < 100
        if not is_action:
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
