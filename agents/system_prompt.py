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
        "Si el usuario mezcla varias acciones en un mensaje (peso + gasto + hábito), ejecuta una tool por acción.\n"
        "Mensaje mixto (emoción + comida + finanzas + ejercicio en un párrafo): NO cambies de agente ni derives. "
        "Orden: (1) create_journal_entry con el texto completo o fiel, (2) log_meal si describe qué comió, "
        "(3) create_transaction / log_exercise / log_habit_completion según detectes datos concretos, "
        "(4) responde en un párrafo breve.\n"
        "Comida: log_meal = qué comió (texto); create_transaction = gasto en euros/francos.\n"
        "Para peso: log_weight con weight_kg, o log_health/update_health con physical.weight_kg (no uses 'weight' solo).\n"
        "Para medidas corporales (cintura, grasa, etc.): log_body_measurement.\n"
        "Si el mensaje tiene contenido emocional, reflexivo o narrativo (cómo te sientes, qué te ronda la cabeza), "
        "usa create_journal_entry con el texto fiel antes de asesorar; luego registra datos operativos si los hay.\n"
        "Para crear un objetivo (goal) necesitas el desire_id; si no lo tienes, usa "
        "get_desire_structure o get_all_desires_full primero.\n"
        "No menciones datos que no hayas obtenido con herramientas."
    )
