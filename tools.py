from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from atlas_client import (
    _post,
    apply_day_template,
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
    get_applied_day_template,
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
    list_day_templates,
    log_exercise,
    log_health,
    log_habit,
    log_relationship,
    log_self_relationship,
    remove_day_template,
    update_desire,
    update_goal,
    update_habit,
    update_health,
    update_relationship,
    update_task,
)

_ZURICH_TZ = ZoneInfo("Europe/Zurich")


def _normalize_habit_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return re.sub(r"\s+", " ", "".join(ch for ch in normalized if not unicodedata.combining(ch))).strip()


def _normalize_habit_log_date(raw: Any) -> str:
    if raw in (None, ""):
        return datetime.now(_ZURICH_TZ).date().isoformat()
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in {"hoy", "today"}:
        return datetime.now(_ZURICH_TZ).date().isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return datetime.now(_ZURICH_TZ).date().isoformat()


def _match_habit_by_title(habits: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    normalized_title = _normalize_habit_text(title)
    if not normalized_title:
        return None
    exact = [habit for habit in habits if _normalize_habit_text(str(habit.get("title") or "")) == normalized_title]
    if len(exact) == 1:
        return exact[0]
    partial = [
        habit
        for habit in habits
        if normalized_title in _normalize_habit_text(str(habit.get("title") or ""))
        or _normalize_habit_text(str(habit.get("title") or "")) in normalized_title
    ]
    if len(partial) == 1:
        return partial[0]
    return None


async def _resolve_habit_for_log(args: dict[str, Any]) -> int:
    payload = await get_today()
    habits = payload.get("habits") if isinstance(payload, dict) else None
    if not isinstance(habits, list) or not habits:
        raise ValueError("No hay hábitos activos hoy en Atlas Vital.")

    habit_title = str(args.get("habit_title") or args.get("title") or "").strip()
    if habit_title:
        matched = _match_habit_by_title(habits, habit_title)
        if matched and matched.get("id") is not None:
            return int(matched["id"])
        options = ", ".join(f"[{habit.get('id')}] {habit.get('title')}" for habit in habits[:8])
        raise ValueError(f"No encontré el hábito '{habit_title}'. Hoy: {options}")

    raw_id = args.get("habit_id")
    if raw_id is None:
        options = ", ".join(f"[{habit.get('id')}] {habit.get('title')}" for habit in habits[:8])
        raise ValueError(f"Falta habit_id o habit_title. Hábitos de hoy: {options}")

    try:
        habit_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"habit_id inválido: {raw_id!r}") from exc

    for habit in habits:
        if habit.get("id") == habit_id:
            return habit_id

    options = ", ".join(f"[{habit.get('id')}] {habit.get('title')}" for habit in habits[:8])
    raise ValueError(f"habit_id {habit_id} no corresponde a un hábito de hoy. Disponibles: {options}")


async def _resolve_template_for_apply(args: dict[str, Any]) -> int:
    payload = await list_day_templates()
    templates = payload.get("templates") if isinstance(payload, dict) else None
    if not isinstance(templates, list) or not templates:
        raise ValueError("No hay plantillas de día en Atlas Vital.")

    template_name = str(args.get("template_name") or args.get("name") or "").strip()
    if template_name:
        normalized_name = _normalize_habit_text(template_name)
        exact = [
            template
            for template in templates
            if _normalize_habit_text(str(template.get("name") or "")) == normalized_name
        ]
        if len(exact) == 1 and exact[0].get("id") is not None:
            return int(exact[0]["id"])
        partial = [
            template
            for template in templates
            if normalized_name in _normalize_habit_text(str(template.get("name") or ""))
            or _normalize_habit_text(str(template.get("name") or "")) in normalized_name
        ]
        if len(partial) == 1 and partial[0].get("id") is not None:
            return int(partial[0]["id"])
        options = ", ".join(f"[{template.get('id')}] {template.get('name')}" for template in templates[:8])
        raise ValueError(f"No encontré la plantilla '{template_name}'. Disponibles: {options}")

    raw_id = args.get("template_id")
    if raw_id is None:
        options = ", ".join(f"[{template.get('id')}] {template.get('name')}" for template in templates[:8])
        raise ValueError(f"Falta template_id o template_name. Plantillas: {options}")

    try:
        template_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"template_id inválido: {raw_id!r}") from exc

    for template in templates:
        if template.get("id") == template_id:
            return template_id

    options = ", ".join(f"[{template.get('id')}] {template.get('name')}" for template in templates[:8])
    raise ValueError(f"template_id {template_id} no existe. Disponibles: {options}")


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


ATLAS_TOOLS: list[dict[str, Any]] = [
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
        "list_day_templates",
        "Lista plantillas de día disponibles (bloques horarios reutilizables).",
        {},
        [],
    ),
    _tool(
        "get_applied_day_template",
        "Consulta qué plantilla está aplicada a una fecha.",
        {
            "date": {"type": "string", "description": "YYYY-MM-DD; omitir = hoy (Zurich)"},
        },
        [],
    ),
    _tool(
        "apply_day_template",
        "Aplica una plantilla de día a una fecha (crea bloques en la agenda).",
        {
            "template_id": {"type": "integer", "description": "ID de list_day_templates"},
            "template_name": {"type": "string", "description": "Nombre de la plantilla si no conoces el ID"},
            "date": {"type": "string", "description": "YYYY-MM-DD; omitir o 'hoy' = hoy (Zurich)"},
        },
        [],
    ),
    _tool(
        "remove_day_template",
        "Quita la plantilla aplicada de una fecha y sus bloques de agenda generados.",
        {
            "date": {"type": "string", "description": "YYYY-MM-DD; omitir o 'hoy' = hoy (Zurich)"},
        },
        [],
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
            "success_criteria": {
                "type": "string",
                "description": "Texto orientativo; puede no persistir en API",
            },
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
            "start_time": {"type": "string", "description": "HH:MM"},
            "end_time": {"type": "string", "description": "HH:MM"},
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
            "start_time": {"type": "string"},
            "end_time": {"type": "string"},
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
        "Marca progreso de un hábito. Llama get_today antes si no tienes el ID exacto, o pasa habit_title.",
        {
            "habit_id": {"type": "integer", "description": "ID exacto de get_today → habits[].id"},
            "habit_title": {"type": "string", "description": "Nombre del hábito si no conoces el ID"},
            "date": {"type": "string", "description": "YYYY-MM-DD; omitir o 'hoy' = fecha actual (Zurich)"},
            "completed": {"type": "boolean"},
            "note": {"type": "string"},
        },
        ["completed"],
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


def _omit_keys(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in keys}


async def dispatch_atlas_tool(name: str, raw: dict[str, Any]) -> Any:
    """Ejecuta una herramienta Atlas según nombre e input JSON del modelo."""
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
    if name == "list_day_templates":
        return await list_day_templates()
    if name == "get_applied_day_template":
        return await get_applied_day_template(_normalize_habit_log_date(args.get("date")))
    if name == "apply_day_template":
        template_id = await _resolve_template_for_apply(args)
        return await apply_day_template(
            template_id=template_id,
            date=_normalize_habit_log_date(args.get("date")),
        )
    if name == "remove_day_template":
        return await remove_day_template(_normalize_habit_log_date(args.get("date")))
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
        if args.get("start_time") is not None:
            extra["start_time"] = args["start_time"]
        if args.get("end_time") is not None:
            extra["end_time"] = args["end_time"]
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
        habit_id = await _resolve_habit_for_log(args)
        return await log_habit(
            habit_id=habit_id,
            date=_normalize_habit_log_date(args.get("date")),
            completed=bool(args.get("completed", True)),
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
