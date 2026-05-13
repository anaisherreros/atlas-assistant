from __future__ import annotations

from .base import Agent
from .prompt_loader import load_agent_prompt

AGENT = Agent(
    name="Asistente personal",
    description="Coordinador general con acceso completo a Atlas Vital.",
    system_prompt=load_agent_prompt("personal"),
)
