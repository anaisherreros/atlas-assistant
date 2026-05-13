from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "agents"


@lru_cache(maxsize=None)
def load_agent_prompt(prompt_name: str) -> str:
    prompt_path = PROMPTS_DIR / f"{prompt_name}.md"
    return prompt_path.read_text(encoding="utf-8").strip()
