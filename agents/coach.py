from __future__ import annotations

from .base import Agent
from .prompt_loader import load_agent_prompt

AGENT = Agent(
    name="Coach",
    description="Claridad de propósito, patrones mentales y crecimiento personal con profundidad práctica.",
    system_prompt=load_agent_prompt("coach"),
)
