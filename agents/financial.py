from __future__ import annotations

from .base import Agent
from .prompt_loader import load_agent_prompt

AGENT = Agent(
    name="Asesor financiero",
    description="Análisis financiero profundo, abundancia e inversión con criterio.",
    system_prompt=load_agent_prompt("financial"),
)
