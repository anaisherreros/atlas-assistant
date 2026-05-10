from __future__ import annotations

import logging
import os
import json
import re
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from atlas_client import (
    _post,
    complete_task,
    create_daily_review,
    create_desire,
    create_goal,
    create_habit,
    create_monthly_review,
    create_patrimony_snapshot,
    create_relationship,
    create_task,
    create_transaction,
    create_weekly_review,
    delete_desire,
    delete_goal,
    delete_habit,
    delete_task,
    delete_transaction,
    get_all_desires_full,
    get_areas_full,
    get_calendar,
    get_dashboard,
    get_desire_structure,
    get_finance,
    get_finance_full,
    get_relationships_full,
    get_reviews_summary,
    get_tasks_pending,
    get_today,
    log_exercise,
    log_health,
    log_habit,
    log_relationship,
    log_self_relationship,
    update_desire,
    update_goal,
    update_habit,
    update_health,
    update_relationship,
    update_task,
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

MODEL = "qwen2.5:14b"
MAX_HISTORY_MESSAGES = 20
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_TOOL_LOOPS = 12


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


ATLAS_OPENAI_TOOLS: list[dict[str, Any]] = [
    _tool(
        "get_today",
        "Obtiene tareas y hábitos de hoy en Atlas Vital.",
        {},
        [],
    ),
    _tool(
        "get_dashboard",
        "Obtiene el dashboard completo (resumen amplio de Atlas Vital).",
        {},
        [],
    ),
    _tool(
        "get_all_desires_full",
        "Lista todos los deseos activos con estructura anidada.",
        {},
        [],
    ),
    _tool(
        "get_desire_structure",
        "Estructura completa de un deseo: objetivos, hábitos y datos asociados.",
        {"desire_id": {"type": "integer", "description": "ID del deseo"}},
        ["desire_id"],
    ),
    _tool(
        "get_areas_full",
        "Todas las áreas y subáreas con IDs y slugs.",
        {},
        [],
    ),
    _tool(
        "get_relationships_full",
        "Relaciones personales con historial reciente.",
        {},
        [],
    ),
    _tool(
        "get_reviews_summary",
        "Resumen de últimas revisiones diaria, semanal, mensual y anual.",
        {},
        [],
    ),
    _tool(
        "get_finance_full",
        "Presupuesto anual completo con categorías y gastos del mes.",
        {},
        [],
    ),
    _tool(
        "get_finance",
        "Detalle financiero del mes en curso: transacciones, totales y categorías.",
        {},
        [],
    ),
    _tool(
        "get_calendar",
        "Hábitos y tareas entre dos fechas (YYYY-MM-DD).",
        {
            "start_date": {"type": "string", "description": "Inicio (YYYY-MM-DD)"},
            "end_date": {"type": "string", "description": "Fin (YYYY-MM-DD)"},
        },
        ["start_date", "end_date"],
    ),
    _tool(
        "get_tasks_pending",
        "Lista de tareas pendientes en Atlas Vital.",
        {},
        [],
    ),
    _tool(
        "create_desire",
        "Crea un nuevo deseo.",
        {
            "title": {"type": "string"},
            "description": {"type": "string", "description": "Opcional"},
            "area": {"type": "string", "description": "Slug o nombre de área, opcional"},
        },
        ["title"],
    ),
    _tool(
        "update_desire",
        "Actualiza un deseo existente (solo envía campos a cambiar).",
        {
            "desire_id": {"type": "integer"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "status": {"type": "string"},
            "priority": {"type": "string"},
        },
        ["desire_id"],
    ),
    _tool(
        "delete_desire",
        "Elimina un deseo.",
        {"desire_id": {"type": "integer"}},
        ["desire_id"],
    ),
    _tool(
        "create_goal",
        "Crea un objetivo asociado a un deseo (desire_id).",
        {
            "desire_id": {"type": "integer"},
            "title": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "success_criteria": {"type": "string", "description": "Texto orientativo; puede no persistir en API"},
        },
        ["desire_id", "title", "start_date", "end_date"],
    ),
    _tool(
        "update_goal",
        "Actualiza un objetivo.",
        {
            "goal_id": {"type": "integer"},
            "title": {"type": "string"},
            "status": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
        },
        ["goal_id"],
    ),
    _tool(
        "delete_goal",
        "Elimina un objetivo.",
        {"goal_id": {"type": "integer"}},
        ["goal_id"],
    ),
    _tool(
        "create_habit",
        "Crea un hábito.",
        {
            "title": {"type": "string"},
            "start_date": {"type": "string"},
            "frequency_type": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly"],
            },
            "goal_id": {"type": "integer", "description": "Opcional"},
            "times_per_period": {"type": "integer"},
            "weekdays": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Días 0-6 si aplica",
            },
        },
        ["title", "start_date", "frequency_type"],
    ),
    _tool(
        "update_habit",
        "Actualiza un hábito.",
        {
            "habit_id": {"type": "integer"},
            "title": {"type": "string"},
            "status": {"type": "string"},
            "frequency_type": {"type": "string"},
            "start_date": {"type": "string"},
            "target_end_date": {"type": "string"},
        },
        ["habit_id"],
    ),
    _tool(
        "delete_habit",
        "Elimina un hábito.",
        {"habit_id": {"type": "integer"}},
        ["habit_id"],
    ),
    _tool(
        "log_habit_completion",
        "Marca progreso de un hábito en una fecha.",
        {
            "habit_id": {"type": "integer"},
            "date": {"type": "string"},
            "completed": {"type": "boolean"},
            "note": {"type": "string"},
        },
        ["habit_id", "date", "completed"],
    ),
    _tool(
        "create_task",
        "Crea una tarea.",
        {
            "title": {"type": "string"},
            "due_date": {"type": "string"},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            "description": {"type": "string"},
            "start_time": {"type": "string", "description": "HH:MM"},
            "end_time": {"type": "string", "description": "HH:MM"},
            "goal_id": {"type": "integer"},
        },
        ["title", "due_date"],
    ),
    _tool(
        "update_task",
        "Actualiza una tarea.",
        {
            "task_id": {"type": "integer"},
            "title": {"type": "string"},
            "due_date": {"type": "string"},
            "priority": {"type": "string"},
            "status": {"type": "string"},
            "start_time": {"type": "string"},
            "end_time": {"type": "string"},
        },
        ["task_id"],
    ),
    _tool(
        "complete_task",
        "Marca una tarea como completada.",
        {"task_id": {"type": "integer"}},
        ["task_id"],
    ),
    _tool(
        "delete_task",
        "Elimina una tarea.",
        {"task_id": {"type": "integer"}},
        ["task_id"],
    ),
    _tool(
        "log_health",
        "Registra datos de salud (objetos physical/emotional/mental como en Atlas).",
        {
            "date": {"type": "string"},
            "physical": {"type": "object", "description": "Opcional: peso, pasos, etc."},
            "emotional": {"type": "object"},
            "mental": {"type": "object"},
        },
        ["date"],
    ),
    _tool(
        "update_health",
        "Actualiza puntuaciones simples de salud para una fecha.",
        {
            "date": {"type": "string"},
            "physical": {"type": "integer"},
            "emotional": {"type": "integer"},
            "mental": {"type": "integer"},
        },
        ["date"],
    ),
    _tool(
        "log_exercise",
        "Registra una sesión de ejercicio.",
        {
            "date": {"type": "string"},
            "exercise_type": {"type": "string"},
            "duration_minutes": {"type": "integer"},
            "note": {"type": "string"},
        },
        ["date", "exercise_type", "duration_minutes"],
    ),
    _tool(
        "create_transaction",
        "Registra una transacción financiera.",
        {
            "description": {"type": "string"},
            "amount": {"type": "number"},
            "transaction_type": {"type": "string", "enum": ["income", "expense"]},
            "date": {"type": "string"},
            "category_id": {"type": "integer", "description": "Opcional"},
        },
        ["description", "amount", "transaction_type", "date"],
    ),
    _tool(
        "delete_transaction",
        "Elimina una transacción por ID.",
        {"transaction_id": {"type": "integer"}},
        ["transaction_id"],
    ),
    _tool(
        "create_patrimony_snapshot",
        "Crea un snapshot de patrimonio; accounts son campos extra según API Atlas.",
        {
            "date": {"type": "string"},
            "accounts": {"type": "object", "description": "Objeto JSON con cuentas / totales"},
        },
        ["date"],
    ),
    _tool(
        "create_relationship",
        "Crea una relación personal.",
        {
            "name": {"type": "string"},
            "relationship_type": {"type": "string"},
            "notes": {"type": "string"},
        },
        ["name", "relationship_type"],
    ),
    _tool(
        "update_relationship",
        "Actualiza datos de una persona en relaciones (person_id).",
        {
            "person_id": {"type": "integer"},
            "name": {"type": "string"},
            "relationship_type": {"type": "string"},
            "notes": {"type": "string"},
        },
        ["person_id"],
    ),
    _tool(
        "log_relationship",
        "Registra una interacción con una persona.",
        {
            "person_id": {"type": "integer"},
            "date": {"type": "string"},
            "interaction_summary": {"type": "string"},
            "feeling": {"type": "string"},
            "note": {"type": "string"},
        },
        ["person_id", "date", "interaction_summary", "feeling"],
    ),
    _tool(
        "log_self_relationship",
        "Registra reflexión sobre la relación contigo misma.",
        {
            "date": {"type": "string"},
            "self_feeling": {"type": "string"},
            "things_i_like": {"type": "string"},
            "working_on": {"type": "string"},
            "note": {"type": "string"},
        },
        ["date", "self_feeling"],
    ),
    _tool(
        "create_daily_review",
        "Crea revisión diaria.",
        {
            "date": {"type": "string"},
            "day_score": {"type": "integer"},
            "mood": {"type": "string"},
            "note": {"type": "string"},
        },
        ["date"],
    ),
    _tool(
        "create_weekly_review",
        "Crea revisión semanal.",
        {
            "week_start": {"type": "string"},
            "week_end": {"type": "string"},
            "what_went_well": {"type": "string"},
            "what_was_hard": {"type": "string"},
            "energy_score": {"type": "integer"},
        },
        ["week_start", "week_end"],
    ),
    _tool(
        "create_monthly_review",
        "Crea revisión mensual.",
        {
            "year": {"type": "integer"},
            "month": {"type": "integer"},
            "financial_review": {"type": "string"},
            "areas_review": {"type": "string"},
        },
        ["year", "month"],
    ),
]


def build_system_prompt(dashboard_data: str) -> str:
    return (
        "Eres el asistente personal de Anaïs.\n"
        "Eres su mano derecha, directa y práctica.\n\n"
        "CONTEXTO ACTUAL DE SU VIDA (datos reales de Atlas Vital, incluidos en este mensaje):\n"
        f"{dashboard_data}\n\n"
        "Usa estos datos como base, pero las herramientas (function calling) te permiten "
        "leer y modificar Atlas Vital en tiempo real.\n"
        "Cuando el usuario pida crear, actualizar, borrar o consultar datos concretos, "
        "usa la herramienta adecuada, ejecuta y confirma con precisión qué ocurrió.\n\n"
        "Para crear un objetivo (goal) necesitas el desire_id; si no lo tienes, usa "
        "get_desire_structure o get_all_desires_full.\n\n"
        "Si los datos del contexto están vacíos en algún área, los puedes completar con las "
        "herramientas de lectura.\n\n"
        "Si la consulta es solo sobre hoy o el día, prioriza get_today o el bloque JSON del "
        "contexto cuando ya refleje el día."
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
        "objetivos del deseo",
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


def _omit_keys(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in keys}


async def dispatch_atlas_tool(name: str, raw: dict[str, Any]) -> Any:
    """Ejecuta una herramienta Atlas según el nombre OpenAI y los argumentos JSON."""
    args = dict(raw)

    if name == "get_today":
        return await get_today()
    if name == "get_dashboard":
        return await get_dashboard()
    if name == "get_all_desires_full":
        return await get_all_desires_full()
    if name == "get_desire_structure":
        return await get_desire_structure(int(args["desire_id"]))
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
    if name == "get_calendar":
        return await get_calendar(args["start_date"], args["end_date"])
    if name == "get_tasks_pending":
        return await get_tasks_pending()

    if name == "create_desire":
        return await create_desire(
            title=args["title"],
            description=args.get("description", ""),
            area=args.get("area", ""),
        )
    if name == "update_desire":
        did = int(args.pop("desire_id"))
        return await update_desire(did, **_omit_keys(args, "desire_id"))
    if name == "delete_desire":
        return await delete_desire(int(args["desire_id"]))

    if name == "create_goal":
        return await create_goal(
            desire_id=int(args["desire_id"]),
            title=args["title"],
            start_date=args["start_date"],
            end_date=args["end_date"],
        )
    if name == "update_goal":
        gid = int(args.pop("goal_id"))
        return await update_goal(gid, **_omit_keys(args, "goal_id"))
    if name == "delete_goal":
        return await delete_goal(int(args["goal_id"]))

    if name == "create_habit":
        extra: dict[str, Any] = {}
        if args.get("times_per_period") is not None:
            extra["times_per_period"] = args["times_per_period"]
        if args.get("weekdays") is not None:
            extra["weekdays"] = args["weekdays"]
        return await create_habit(
            title=args["title"],
            start_date=args["start_date"],
            frequency_type=args.get("frequency_type", "daily"),
            goal_id=args.get("goal_id"),
            **extra,
        )
    if name == "update_habit":
        hid = int(args.pop("habit_id"))
        return await update_habit(hid, **_omit_keys(args, "habit_id"))
    if name == "delete_habit":
        return await delete_habit(int(args["habit_id"]))
    if name == "log_habit_completion":
        return await log_habit(
            habit_id=int(args["habit_id"]),
            date=args["date"],
            completed=bool(args["completed"]),
            note=args.get("note", ""),
        )

    if name == "create_task":
        return await create_task(
            title=args["title"],
            due_date=args["due_date"],
            description=args.get("description", ""),
            priority=args.get("priority", "medium"),
            start_time=args.get("start_time"),
            end_time=args.get("end_time"),
            goal_id=args.get("goal_id"),
        )
    if name == "update_task":
        tid = int(args.pop("task_id"))
        return await update_task(tid, **_omit_keys(args, "task_id"))
    if name == "complete_task":
        return await complete_task(int(args["task_id"]))
    if name == "delete_task":
        return await delete_task(int(args["task_id"]))

    if name == "log_health":
        return await log_health(
            date=args["date"],
            physical=args.get("physical"),
            emotional=args.get("emotional"),
            mental=args.get("mental"),
        )
    if name == "update_health":
        return await update_health(
            date=args["date"],
            physical=args.get("physical"),
            emotional=args.get("emotional"),
            mental=args.get("mental"),
        )
    if name == "log_exercise":
        return await log_exercise(
            date=args["date"],
            exercise_type=args["exercise_type"],
            duration_minutes=int(args["duration_minutes"]),
            note=args.get("note", ""),
        )

    if name == "create_transaction":
        payload: dict[str, Any] = {
            "description": args["description"],
            "amount": float(args["amount"]),
            "transaction_type": args["transaction_type"],
            "date": args["date"],
        }
        if args.get("category_id") is not None:
            payload["category_id"] = int(args["category_id"])
            return await _post("/api/assistant/finance/transaction/", payload)
        return await create_transaction(
            description=payload["description"],
            amount=payload["amount"],
            transaction_type=payload["transaction_type"],
            date=payload["date"],
        )
    if name == "delete_transaction":
        return await delete_transaction(int(args["transaction_id"]))
    if name == "create_patrimony_snapshot":
        d = args["date"]
        ac = args.get("accounts")
        if isinstance(ac, dict):
            return await create_patrimony_snapshot(d, **ac)
        return await create_patrimony_snapshot(d)

    if name == "create_relationship":
        return await create_relationship(
            name=args["name"],
            relationship_type=args["relationship_type"],
            notes=args.get("notes", ""),
        )
    if name == "update_relationship":
        return await update_relationship(
            person_id=int(args["person_id"]),
            name=args.get("name"),
            relationship_type=args.get("relationship_type"),
            notes=args.get("notes"),
        )
    if name == "log_relationship":
        return await log_relationship(
            person_id=int(args["person_id"]),
            date=args["date"],
            interaction_summary=args["interaction_summary"],
            feeling=args["feeling"],
            note=args.get("note", ""),
        )
    if name == "log_self_relationship":
        note = (args.get("note") or "").strip()
        things = args.get("things_i_like") or ""
        if note:
            things = f"{things}\n\nNota: {note}".strip() if things else f"Nota: {note}"
        return await log_self_relationship(
            date=args["date"],
            self_feeling=args["self_feeling"],
            things_i_like=things,
            working_on=args.get("working_on", ""),
        )

    if name == "create_daily_review":
        return await create_daily_review(
            date=args["date"],
            day_score=args.get("day_score"),
            mood=args.get("mood", ""),
            note=args.get("note", ""),
        )
    if name == "create_weekly_review":
        ws, we = args["week_start"], args["week_end"]
        rest = _omit_keys(args, "week_start", "week_end")
        return await create_weekly_review(ws, we, **rest)
    if name == "create_monthly_review":
        y, m = int(args["year"]), int(args["month"])
        rest = _omit_keys(args, "year", "month")
        return await create_monthly_review(y, m, **rest)

    raise ValueError(f"Herramienta no reconocida: {name}")


async def generate_with_tools(
    client: AsyncOpenAI,
    *,
    model: str,
    system_prompt: str,
    api_messages: list[dict[str, Any]],
) -> tuple[str, bool]:
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *api_messages,
    ]
    tools_were_used = False

    for loop_idx in range(MAX_TOOL_LOOPS):
        logger.info(
            "Llamada modelo [%s] con %d mensajes (tools habilitadas)",
            loop_idx + 1,
            len(conversation),
        )
        response = await client.chat.completions.create(
            model=model,
            messages=conversation,
            tools=ATLAS_OPENAI_TOOLS,
            tool_choice="auto",
            max_tokens=8192,
        )
        usage = response.usage
        if usage is not None:
            logger.info(
                "Tokens - prompt: %s completion: %s",
                usage.prompt_tokens,
                usage.completion_tokens,
            )

        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason
        logger.info("Finish reason: %s", finish)

        if not msg.tool_calls:
            text = (msg.content or "").strip()
            return (
                text or "(Sin contenido de texto en la respuesta.)",
                tools_were_used,
            )

        tools_were_used = True
        assistant_entry: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ],
        }
        conversation.append(assistant_entry)

        for tc in msg.tool_calls:
            fname = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                parsed: dict[str, Any] = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning("JSON inválido en tool %s: %s", fname, raw_args)
                parsed = {}
            logger.info("Ejecutando tool %s con %s", fname, parsed)
            try:
                result = await dispatch_atlas_tool(fname, parsed)
                out = json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                logger.exception("Error en tool %s", fname)
                out = json.dumps({"error": str(exc)}, ensure_ascii=False)

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": out,
                }
            )

    return (
        "Alcanzado el límite de pasadas de herramientas sin respuesta final.",
        tools_were_used,
    )


async def post_init(application: Application) -> None:
    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    await init_db(engine)
    application.bot_data["engine"] = engine
    application.bot_data["session_factory"] = session_factory(engine)
    application.bot_data["openai"] = AsyncOpenAI(
        base_url="http://46.224.210.224:11434/v1",
        api_key="ollama",
    )
    logger.info("Base de datos lista y cliente OpenAI-compatible (Ollama) configurado.")


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
    client: AsyncOpenAI = context.application.bot_data["openai"]

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
        logger.info("Historial chat: limit=%s (mensajes=%d)", history_limit, len(api_messages))

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
            model = MODEL
            logger.info("Clasificacion de mensaje: %s (modelo: %s)", complexity, model)
            logger.info("Modelo elegido: %s para: %s", model, text[:50])
            assistant_text, tools_used = await generate_with_tools(
                client,
                model=model,
                system_prompt=build_system_prompt(dashboard_data),
                api_messages=api_messages,
            )
        except Exception:
            logger.exception("Error al llamar a la API de Ollama (OpenAI-compatible)")
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
