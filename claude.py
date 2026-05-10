from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic

from tools import ATLAS_TOOLS, dispatch_atlas_tool

logger = logging.getLogger(__name__)


def _serialize_assistant_content(content: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return blocks


async def generate_with_tools(
    client: AsyncAnthropic,
    *,
    model: str,
    system_prompt: str,
    api_messages: list[dict[str, Any]],
    max_tool_loops: int = 12,
) -> tuple[str, bool]:
    conversation_messages: list[dict[str, Any]] = list(api_messages)
    tools_were_used = False

    for loop_idx in range(max_tool_loops):
        logger.info(
            "Llamada Claude [%s] con %d mensajes (tools habilitadas)",
            loop_idx + 1,
            len(conversation_messages),
        )
        response = await client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            tools=ATLAS_TOOLS,
            messages=conversation_messages,
        )
        logger.info(
            "Tokens - input: %s output: %s",
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        logger.info("Stop reason: %s", response.stop_reason)

        assistant_text_parts: list[str] = []
        assistant_content = _serialize_assistant_content(list(response.content))
        tool_result_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                assistant_text_parts.append(block.text)
                continue
            if block.type != "tool_use":
                continue

            tools_were_used = True
            tool_input = block.input if isinstance(block.input, dict) else {}
            logger.info("Ejecutando tool: %s con input: %s", block.name, tool_input)

            try:
                result = await dispatch_atlas_tool(block.name, tool_input)
                logger.info("Resultado de tool %s: %s", block.name, result)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            except Exception as exc:
                logger.exception("Error ejecutando tool de Atlas Vital: %s", block.name)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error al ejecutar {block.name}: {exc}",
                        "is_error": True,
                    }
                )

        conversation_messages.append(
            {"role": "assistant", "content": assistant_content}
        )

        if tool_result_blocks:
            conversation_messages.append(
                {"role": "user", "content": tool_result_blocks}
            )
            continue

        assistant_text = "".join(assistant_text_parts).strip()
        if assistant_text:
            return assistant_text, tools_were_used
        return "(Sin contenido de texto en la respuesta.)", tools_were_used

    return (
        "Alcanzado el límite de pasadas de herramientas sin respuesta final.",
        tools_were_used,
    )
