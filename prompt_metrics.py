from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Aproximación rápida (~4 chars/token en español).
_CHARS_PER_TOKEN = 4

# Referencia medida en repo (jun 2026): system prompts completos con bloque Atlas.
BASELINE_SYSTEM_TOKENS: dict[str, int] = {
    "personal": 1430,
    "coach": 1963,
    "financial": 1087,
    "performance": 1607,
}
BASELINE_MEMORY_WRAPPER_CHARS = 120
BASELINE_MEMORY_WRAPPER_TOKENS = 30


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += estimate_tokens(str(block.get("text", "")))
                    elif block.get("type") == "tool_result":
                        total += estimate_tokens(str(block.get("content", "")))
    return total


def log_prompt_footprint(
    *,
    chat_id: int,
    agent_key: str,
    model: str,
    tier_reason: str,
    system_prompt: str,
    api_messages: list[dict[str, Any]],
    memory_chars: int = 0,
) -> None:
    system_tokens = estimate_tokens(system_prompt)
    history_tokens = estimate_messages_tokens(api_messages)
    memory_tokens = (memory_chars // _CHARS_PER_TOKEN) if memory_chars else 0
    baseline = BASELINE_SYSTEM_TOKENS.get(agent_key, 0)
    delta = system_tokens - baseline if baseline else 0

    logger.info(
        "Prompt footprint chat=%s agent=%s model=%s tier=%s | "
        "system~%d tok (baseline~%d, %+d) history~%d tok memory~%d tok total~%d tok",
        chat_id,
        agent_key,
        model,
        tier_reason,
        system_tokens,
        baseline,
        delta,
        history_tokens,
        memory_tokens,
        system_tokens + history_tokens,
    )
