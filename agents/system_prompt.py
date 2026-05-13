from __future__ import annotations

from .base import Agent


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
