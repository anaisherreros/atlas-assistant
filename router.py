from __future__ import annotations

from agents.registry import AGENT_ALIASES

VALID_AGENT_KEYS = frozenset({"personal", "coach", "performance", "financial"})

_EXPLICIT_SWITCH_MARKERS = (
    "pasame con",
    "pásame con",
    "habla con",
    "quiero hablar con",
    "cambiar a",
    "cambiar al",
    "cambiar a la",
    "cambiar al",
    "pasar a",
    "pasar al",
    "pasar a la",
)


def detect_agent(message: str, current_agent: str) -> str:
    msg = message.lower()
    current_agent = AGENT_ALIASES.get(current_agent, current_agent)
    if current_agent not in VALID_AGENT_KEYS:
        current_agent = "personal"

    if not any(marker in msg for marker in _EXPLICIT_SWITCH_MARKERS):
        return current_agent

    if "coach" in msg:
        return "coach"
    if "nutricion" in msg or "nutricionista" in msg:
        return "performance"
    if "entrenador" in msg or "entrenadora" in msg:
        return "performance"
    if "rendimiento" in msg or "salud" in msg:
        return "performance"
    if "finanzas" in msg or "financiero" in msg or "financiera" in msg:
        return "financial"
    if "asistente" in msg or "personal" in msg or "vuelve" in msg:
        return "personal"

    return current_agent
