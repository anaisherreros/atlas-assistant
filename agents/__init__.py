from __future__ import annotations

import json
import logging
from typing import Any

from atlas_client import (
    get_dashboard,
    get_exercise_recent,
    get_finance_full,
    get_health_context,
    get_health_emotional,
    get_reviews_summary,
)

from .base import Agent
from . import coach, nutritionist, personal, trainer

logger = logging.getLogger(__name__)

DEFAULT_AGENT_KEY = "personal"

AGENTS: dict[str, Agent] = {
    "personal": personal.AGENT,
    "coach": coach.AGENT,
    "nutritionist": nutritionist.AGENT,
    "trainer": trainer.AGENT,
}


def build_agent_system_prompt(agent: Agent, context_data: str) -> str:
    return (
        f"{agent.system_prompt}\n\n"
        "DATOS ACTUALES DE ANAÏS (Atlas Vital):\n"
        f"{context_data}\n\n"
        "Usa estos datos para personalizar tus respuestas.\n"
        "Si un dato no está disponible, no lo menciones.\n\n"
        "Las herramientas (tool use de Anthropic) permiten leer y modificar Atlas Vital "
        "cuando encaje con tu rol y las instrucciones anteriores.\n"
        "Para crear un objetivo (goal) necesitas el desire_id; si no lo tienes, usa "
        "get_desire_structure o get_all_desires_full.\n"
        "Si el bloque JSON está vacío en algún área, puedes completarlo con herramientas "
        "de lectura.\n"
        "Si la consulta es sobre hoy o el día, prioriza get_today o el contexto cuando "
        "ya refleje el día."
    )


def get_agent(agent_key: str) -> Agent:
    return AGENTS.get(agent_key, personal.AGENT)


async def fetch_context_for_agent(agent_key: str) -> str:
    """Carga JSON de contexto Atlas según el agente activo."""
    payload: Any
    try:
        if agent_key == "personal":
            payload = await get_dashboard()
        elif agent_key == "coach":
            payload = {
                "dashboard": await get_dashboard(),
                "reviews_summary": await get_reviews_summary(),
            }
        elif agent_key in ("nutritionist", "trainer"):
            health_today: Any = {}
            health_emotional: Any = None
            exercise_recent: Any = {}
            try:
                health_today = await get_health_context()
            except Exception:
                logger.exception("Fallo get_health_context (health/today)")
            try:
                health_emotional = await get_health_emotional()
            except Exception as exc:
                logger.warning(
                    "get_health_emotional no disponible o error: %s",
                    exc,
                )
                health_emotional = None
            try:
                exercise_recent = await get_exercise_recent(days=7)
            except Exception:
                logger.exception("Fallo get_exercise_recent")
            payload = {
                "health_today": health_today,
                "health_emotional": health_emotional,
                "exercise_recent": exercise_recent,
            }
        elif agent_key == "financial":
            payload = await get_finance_full()
        else:
            payload = await get_dashboard()
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        logger.exception("Error cargando contexto Atlas para agente %s", agent_key)
        return "{}"
