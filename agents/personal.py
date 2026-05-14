from __future__ import annotations

from .base import Agent
from .prompt_loader import load_agent_prompt

AGENT = Agent(
    name="Asistente personal",
    description="Mano derecha estratégica que organiza, prioriza y coordina a los especialistas.",
    system_prompt=load_agent_prompt("personal"),
)
