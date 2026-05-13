from __future__ import annotations

from .base import Agent
from .prompt_loader import load_agent_prompt

AGENT = Agent(
    name="Coach",
    description="Metas, mentalidad y crecimiento personal.",
    system_prompt=load_agent_prompt("coach"),
)
