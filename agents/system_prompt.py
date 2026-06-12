from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .base import Agent

_WEEKDAYS_ES = (
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
)


def _current_datetime_context_block() -> str:
    now = datetime.now(ZoneInfo("Europe/Zurich"))
    weekday = _WEEKDAYS_ES[now.weekday()]
    return (
        f"FECHA Y HORA ACTUAL (Europe/Zurich):\n"
        f"- {weekday}, {now.strftime('%d/%m/%Y')} · {now.strftime('%H:%M')} {now.tzname() or 'CET/CEST'}\n\n"
        "Cuando el usuario diga 'hoy', 'mañana', 'ayer', un día de la semana, "
        "o no especifique fecha, calcula a partir de esta fecha. "
        "Nunca preguntes la fecha si puede inferirse."
    )


def build_agent_system_prompt(agent: Agent) -> str:
    return (
        f"{_current_datetime_context_block()}\n\n"
        f"{agent.system_prompt}\n\n"
        "Tienes herramientas de Atlas Vital para leer y modificar los datos de Anaïs.\n\n"
        "Usa las tools cuando necesites datos actuales (get_today, get_dashboard, etc.) "
        "o cuando tengas que crear, actualizar o eliminar algo.\n"
        "Antes de log_habit_completion, llama get_today y usa habits[].id exacto, "
        "o pasa habit_title con el nombre del hábito. La fecha va en YYYY-MM-DD "
        "(o omítela para hoy).\n"
        "Para plantillas de día: list_day_templates para ver opciones; apply_day_template "
        "con template_name o template_id; remove_day_template para quitarla de una fecha.\n"
        "Para create_transaction: get_finance para categories[] o pasa category_name (comida, ocio, etc.).\n"
        "Para crear un objetivo (goal) necesitas el desire_id; si no lo tienes, usa "
        "get_desire_structure o get_all_desires_full primero.\n"
        "No menciones datos que no hayas obtenido con herramientas."
    )
