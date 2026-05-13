from __future__ import annotations

from .base import Agent
from .prompt_loader import load_agent_prompt

AGENT = Agent(
    name="Especialista en rendimiento",
    description="Salud, nutrición, entrenamiento y recuperación en una sola visión.",
    system_prompt=load_agent_prompt("performance"),
)
