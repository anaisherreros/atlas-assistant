from __future__ import annotations

from .base import Agent
from . import coach, financial, performance, personal

DEFAULT_AGENT_KEY = "personal"
AGENT_ALIASES: dict[str, str] = {
    "nutritionist": "performance",
    "trainer": "performance",
}

AGENTS: dict[str, Agent] = {
    "personal": personal.AGENT,
    "coach": coach.AGENT,
    "financial": financial.AGENT,
    "performance": performance.AGENT,
}


def get_agent(agent_key: str) -> Agent:
    normalized = AGENT_ALIASES.get(agent_key, agent_key)
    return AGENTS.get(normalized, personal.AGENT)
