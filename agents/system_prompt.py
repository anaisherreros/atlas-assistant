from __future__ import annotations

from .base import Agent


def build_agent_system_prompt(agent: Agent) -> str:
    return (
        f"{agent.system_prompt}\n\n"
        "Tienes acceso al servidor MCP de Atlas Vital con todas las herramientas necesarias "
        "para leer y modificar los datos de Anaïs.\n\n"
        "Usa las herramientas cuando necesites datos actuales (get_today, get_dashboard, etc.) "
        "o cuando tengas que crear, actualizar o eliminar algo.\n"
        "Para crear un objetivo (goal) necesitas el desire_id; si no lo tienes, usa "
        "get_desire_structure o get_all_desires_full primero.\n"
        "No menciones datos que no hayas obtenido con herramientas."
    )
