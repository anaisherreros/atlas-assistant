from __future__ import annotations

from .base import Agent
from . import coach, nutritionist, personal, trainer

DEFAULT_AGENT_KEY = "personal"

AGENTS: dict[str, Agent] = {
    "personal": personal.AGENT,
    "coach": coach.AGENT,
    "nutritionist": nutritionist.AGENT,
    "trainer": trainer.AGENT,
}


def build_agent_system_prompt(agent: Agent, dashboard_data: str) -> str:
    return (
        f"{agent.system_prompt}\n\n"
        "CONTEXTO ACTUAL DE ATLAS VITAL (datos reales, JSON):\n"
        f"{dashboard_data}\n\n"
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
