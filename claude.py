from __future__ import annotations

import logging
import os
from typing import Any

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

_MCP_URL = os.environ.get("ATLAS_VITAL_URL", "").rstrip("/") + "/mcp"
_MCP_TOKEN = os.environ.get("ASSISTANT_API_KEY", "")

_MCP_SERVERS = [
    {
        "type": "url",
        "url": _MCP_URL,
        "name": "atlas_vital",
        "authorization_token": _MCP_TOKEN,
    }
]


async def generate_with_tools(
    client: AsyncAnthropic,
    *,
    model: str,
    system_prompt: str,
    api_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,  # ignorado, tools vienen del MCP server
    max_tool_loops: int = 12,  # ignorado, Anthropic lo gestiona internamente
) -> tuple[str, bool]:
    """
    Llama a Claude con el servidor MCP de Atlas Vital.
    Anthropic gestiona el ciclo de tool calls contra el MCP server.
    """
    response = await client.beta.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=api_messages,
        mcp_servers=_MCP_SERVERS,
        betas=["mcp-client-2025-04-04"],
    )

    logger.info(
        "Tokens - input: %s output: %s | stop: %s",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.stop_reason,
    )

    text_parts: list[str] = []
    tools_used = False
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type in ("tool_use", "mcp_tool_use", "mcp_tool_result"):
            tools_used = True

    text = "".join(text_parts).strip()
    return text or "(Sin respuesta de texto.)", tools_used
