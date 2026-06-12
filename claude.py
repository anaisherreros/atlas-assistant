from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from atlas_client import AtlasApiError
from tools import ATLAS_TOOLS, dispatch_atlas_tool

logger = logging.getLogger(__name__)

WRITE_TOOLS = frozenset({
    "log_habit_completion",
    "log_health",
    "log_weight",
    "log_body_measurement",
    "update_health",
    "log_exercise",
    "create_transaction",
    "create_task",
    "complete_task",
    "log_self_relationship",
    "log_relationship",
    "apply_day_template",
    "remove_day_template",
})

_INSPECT_LOGS_REMAINING = int(os.getenv("CLAUDE_INSPECT_LOGS", "50"))


@dataclass
class _WriteToolOutcome:
    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    is_error: bool = False
    error_message: str = ""


def _inspect_log_response_blocks(content: list[Any], *, loop_idx: int, stop_reason: str) -> None:
    global _INSPECT_LOGS_REMAINING
    if _INSPECT_LOGS_REMAINING <= 0:
        return
    _INSPECT_LOGS_REMAINING -= 1
    blocks: list[dict[str, Any]] = []
    for block in content:
        block_type = getattr(block, "type", None)
        entry: dict[str, Any] = {"type": block_type}
        if block_type == "text":
            text = getattr(block, "text", "") or ""
            entry["text"] = text if len(text) <= 300 else text[:300] + "…"
        elif block_type == "tool_use":
            entry["id"] = getattr(block, "id", None)
            entry["name"] = getattr(block, "name", None)
            entry["input"] = getattr(block, "input", None)
        elif block_type == "mcp_tool_use":
            entry["id"] = getattr(block, "id", None)
            entry["name"] = getattr(block, "name", None)
            entry["server_name"] = getattr(block, "server_name", None)
            entry["input"] = getattr(block, "input", None)
        elif block_type == "mcp_tool_result":
            entry["tool_use_id"] = getattr(block, "tool_use_id", None)
            entry["is_error"] = getattr(block, "is_error", None)
            raw = getattr(block, "content", None)
            if isinstance(raw, str):
                entry["content"] = raw if len(raw) <= 400 else raw[:400] + "…"
            else:
                entry["content"] = str(raw)[:400]
        blocks.append(entry)
    logger.warning(
        "INSPECT response blocks loop=%s stop=%s (quedan %s): %s",
        loop_idx + 1,
        stop_reason,
        _INSPECT_LOGS_REMAINING,
        json.dumps(blocks, ensure_ascii=False),
    )


def _parse_json_payload(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    if isinstance(raw, list):
        texts = [
            getattr(item, "text", "")
            for item in raw
            if getattr(item, "type", None) == "text"
        ]
        joined = "".join(texts)
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined
    return raw


def _format_date_label(raw: Any) -> str:
    if not raw:
        return "hoy"
    text = str(raw).strip()
    return text[:10] if len(text) >= 10 else text


def _format_amount(amount: Any) -> str:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    currency = os.getenv("DEFAULT_CURRENCY", "CHF")
    formatted = f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{formatted} {currency}"


def _entity_title(result: dict[str, Any], *keys: str, fallback: str = "elemento") -> str:
    for key in keys:
        node = result.get(key)
        if isinstance(node, dict):
            for field_name in ("title", "name", "description"):
                val = node.get(field_name)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return fallback


def _api_error_message(result: Any) -> str | None:
    if isinstance(result, dict):
        detail = result.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    if isinstance(result, str) and result.strip().lower().startswith("error"):
        return result.strip()
    return None


def _build_write_confirmation(outcome: _WriteToolOutcome) -> str:
    name = outcome.tool_name
    inp = outcome.tool_input
    labels = {
        "log_habit_completion": ("marcar el hábito", "hábito"),
        "log_health": ("registrar salud", "salud"),
        "log_weight": ("registrar el peso", "peso"),
        "log_body_measurement": ("registrar medidas corporales", "medición"),
        "update_health": ("actualizar salud", "salud"),
        "log_exercise": ("registrar ejercicio", "ejercicio"),
        "create_transaction": ("registrar el movimiento", "movimiento"),
        "create_task": ("crear la tarea", "tarea"),
        "complete_task": ("completar la tarea", "tarea"),
        "log_self_relationship": ("registrar reflexión personal", "reflexión"),
        "log_relationship": ("registrar la interacción", "interacción"),
        "apply_day_template": ("aplicar la plantilla", "plantilla"),
        "remove_day_template": ("quitar la plantilla", "plantilla"),
    }
    verb, noun = labels.get(name, (name, name))

    if outcome.is_error:
        return f"❌ No se pudo {verb}: {outcome.error_message or 'error desconocido'}"

    payload = outcome.result if isinstance(outcome.result, dict) else {}
    api_err = _api_error_message(outcome.result)
    if api_err:
        return f"❌ No se pudo {verb}: {api_err}"

    if name == "log_habit_completion":
        title = _entity_title(payload, "habit", fallback="hábito")
        date_label = _format_date_label((payload.get("habit_log") or {}).get("date") or inp.get("date"))
        if inp.get("completed") is False:
            return f"✅ Hábito '{title}' desmarcado · {date_label}"
        return f"✅ Hábito '{title}' marcado · {date_label}"

    if name == "create_task":
        title = _entity_title(payload, "task", fallback=str(inp.get("title") or "tarea"))
        if payload.get("created") is False:
            return f"⚠️ Ya existía: {title}"
        date_label = _format_date_label((payload.get("task") or {}).get("due_date") or inp.get("due_date"))
        return f"✅ Tarea '{title}' creada · {date_label}"

    if name == "complete_task":
        title = _entity_title(payload, "task", fallback="tarea")
        return f"✅ Tarea '{title}' completada"

    if name == "create_transaction":
        tx = payload.get("transaction") if isinstance(payload.get("transaction"), dict) else payload
        desc = (tx or {}).get("description") or inp.get("description") or "movimiento"
        amount = _format_amount((tx or {}).get("amount") or inp.get("amount"))
        tx_type = (tx or {}).get("transaction_type") or inp.get("transaction_type")
        label = "Ingreso" if tx_type == "income" else "Gasto"
        cat = (tx or {}).get("category")
        cat_name = cat.get("name") if isinstance(cat, dict) else None
        line = f"✅ {label}: {desc} · {amount}"
        return f"{line} · {cat_name}" if cat_name else line

    if name == "log_health":
        date_label = _format_date_label(payload.get("date") or inp.get("date"))
        parts = [
            key.replace("_log", "")
            for key in ("physical_log", "emotional_log", "mental_log")
            if payload.get(key)
        ]
        scope = ", ".join(parts) if parts else "salud"
        physical = payload.get("physical_log") if isinstance(payload.get("physical_log"), dict) else {}
        weight = physical.get("weight_kg")
        if weight is not None:
            return f"✅ Peso registrado · {weight} kg · {date_label}"
        return f"✅ Salud registrada · {date_label} ({scope})"

    if name == "log_weight":
        date_label = _format_date_label(inp.get("date"))
        physical = payload.get("physical_log") if isinstance(payload.get("physical_log"), dict) else {}
        weight = physical.get("weight_kg") or inp.get("weight_kg")
        return f"✅ Peso registrado · {weight} kg · {date_label}"

    if name == "log_body_measurement":
        measurement = payload.get("body_measurement") if isinstance(payload.get("body_measurement"), dict) else {}
        date_label = _format_date_label(measurement.get("date") or inp.get("date"))
        weight = measurement.get("weight_kg") or inp.get("weight_kg")
        waist = measurement.get("waist_cm") or inp.get("waist_cm")
        details = []
        if weight is not None:
            details.append(f"{weight} kg")
        if waist is not None:
            details.append(f"cintura {waist} cm")
        summary = " · ".join(details) if details else "medidas guardadas"
        return f"✅ Medición corporal · {summary} · {date_label}"

    if name == "update_health":
        date_label = _format_date_label(payload.get("date") or inp.get("date"))
        physical = payload.get("physical_log") if isinstance(payload.get("physical_log"), dict) else {}
        if physical.get("weight_kg") is not None:
            return f"✅ Peso actualizado · {physical['weight_kg']} kg · {date_label}"
        return f"✅ Salud actualizada · {date_label}"

    if name == "log_exercise":
        log = payload.get("exercise_log") if isinstance(payload.get("exercise_log"), dict) else {}
        ex_type = log.get("exercise_type") or inp.get("exercise_type") or "ejercicio"
        mins = log.get("duration_minutes") or inp.get("duration_minutes")
        date_label = _format_date_label(log.get("date") or inp.get("date"))
        return f"✅ Ejercicio '{ex_type}' · {mins} min · {date_label}"

    if name == "log_relationship":
        date_label = _format_date_label((payload.get("relationship_log") or {}).get("date") or inp.get("date"))
        summary = inp.get("interaction_summary") or "interacción"
        return f"✅ Interacción registrada · {summary} · {date_label}"

    if name == "log_self_relationship":
        log = payload.get("self_relationship_log") if isinstance(payload.get("self_relationship_log"), dict) else {}
        date_label = _format_date_label(log.get("date") or inp.get("date"))
        feeling = inp.get("self_feeling") or "reflexión"
        return f"✅ Reflexión personal registrada · {feeling} · {date_label}"

    if name == "apply_day_template":
        applied = payload.get("applied_template") if isinstance(payload.get("applied_template"), dict) else {}
        template = applied.get("template") if isinstance(applied.get("template"), dict) else {}
        title = template.get("name") or inp.get("template_name") or "plantilla"
        date_label = _format_date_label(payload.get("date") or applied.get("date") or inp.get("date"))
        blocks = payload.get("time_logs_created")
        if blocks is None:
            blocks = payload.get("block_count")
        suffix = f" · {blocks} bloques" if blocks is not None else ""
        if payload.get("created") is False:
            return f"✅ Plantilla '{title}' actualizada · {date_label}{suffix}"
        return f"✅ Plantilla '{title}' aplicada · {date_label}{suffix}"

    if name == "remove_day_template":
        date_label = _format_date_label(payload.get("date") or inp.get("date"))
        if not payload.get("removed"):
            return f"⚠️ No había plantilla aplicada · {date_label}"
        deleted = payload.get("time_logs_deleted")
        suffix = f" · {deleted} bloques quitados" if deleted is not None else ""
        return f"✅ Plantilla quitada · {date_label}{suffix}"

    return f"✅ {noun.capitalize()} registrado"


_WRITE_CLAIM_RE = re.compile(
    r"\b(marqu[eé]|marcad[oa]|registr[eéao]|cre[eéao]|complet[eéao]|apunt[eéao]|guard[eéao]|"
    r"hecho|listo|ya está|ya esta|done)\b",
    re.IGNORECASE,
)


def _model_claims_write_without_evidence(text: str, write_outcomes: list[_WriteToolOutcome]) -> bool:
    if write_outcomes:
        return False
    return bool(_WRITE_CLAIM_RE.search(text or ""))


def _text_redundant_with_confirmations(text: str, confirmations: list[str]) -> bool:
    if not text or not confirmations:
        return False
    normalized = text.strip().lower()
    if normalized in {
        "listo",
        "listo!",
        "listo.",
        "hecho",
        "ok",
        "vale",
        "perfecto",
        "ya está",
        "ya esta",
        "done",
    }:
        return True
    if len(text.split()) <= 5 and _WRITE_CLAIM_RE.search(text):
        return True
    return False


def _exception_message(exc: Exception) -> str:
    if isinstance(exc, AtlasApiError):
        return exc.detail
    return str(exc)


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def _finalize_user_message(model_text: str, write_outcomes: list[_WriteToolOutcome]) -> str:
    confirmations = _dedupe_lines(
        [_build_write_confirmation(outcome) for outcome in write_outcomes]
    )
    confirmations = [line for line in confirmations if line]
    text = (model_text or "").strip()

    if _model_claims_write_without_evidence(text, write_outcomes):
        warning = "⚠️ No ejecuté ninguna acción — ¿quieres que lo haga?"
        if confirmations:
            return "\n".join(confirmations)
        return f"{warning}\n\n{text}" if text else warning

    if not confirmations:
        return text or "(Sin contenido de texto en la respuesta.)"

    if _text_redundant_with_confirmations(text, confirmations):
        return "\n".join(confirmations)

    return f"{chr(10).join(confirmations)}\n\n{text}"


def _collect_mcp_write_outcomes(
    content: list[Any],
    *,
    skip_tool_use_ids: set[str],
) -> list[_WriteToolOutcome]:
    id_to_name: dict[str, str] = {}
    id_to_input: dict[str, dict[str, Any]] = {}
    for block in content:
        if getattr(block, "type", None) != "mcp_tool_use":
            continue
        tool_id = getattr(block, "id", None)
        if not isinstance(tool_id, str):
            continue
        id_to_name[tool_id] = str(getattr(block, "name", "") or "")
        raw_input = getattr(block, "input", None)
        id_to_input[tool_id] = raw_input if isinstance(raw_input, dict) else {}

    outcomes: list[_WriteToolOutcome] = []
    for block in content:
        if getattr(block, "type", None) != "mcp_tool_result":
            continue
        tool_use_id = getattr(block, "tool_use_id", None)
        if not isinstance(tool_use_id, str) or tool_use_id in skip_tool_use_ids:
            continue
        tool_name = id_to_name.get(tool_use_id, "")
        if tool_name not in WRITE_TOOLS:
            continue
        is_error = bool(getattr(block, "is_error", False))
        raw_content = getattr(block, "content", None)
        if is_error:
            outcomes.append(
                _WriteToolOutcome(
                    tool_name=tool_name,
                    tool_input=id_to_input.get(tool_use_id, {}),
                    result=None,
                    is_error=True,
                    error_message=str(_parse_json_payload(raw_content)),
                )
            )
        else:
            outcomes.append(
                _WriteToolOutcome(
                    tool_name=tool_name,
                    tool_input=id_to_input.get(tool_use_id, {}),
                    result=_parse_json_payload(raw_content),
                    is_error=False,
                )
            )
    return outcomes


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
    tools: list[dict[str, Any]] | None = None,
    max_tool_loops: int = 12,
) -> tuple[str, bool]:
    """Claude con tool use local: el bot llama a Atlas Vital vía REST (atlas_client)."""
    conversation_messages: list[dict[str, Any]] = list(api_messages)
    tools_were_used = False
    available_tools = ATLAS_TOOLS if tools is None else tools
    write_outcomes: list[_WriteToolOutcome] = []

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
            tools=available_tools,
            messages=conversation_messages,
        )
        _inspect_log_response_blocks(
            list(response.content),
            loop_idx=loop_idx,
            stop_reason=str(response.stop_reason),
        )
        logger.info(
            "Tokens - input: %s output: %s | stop: %s",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.stop_reason,
        )

        assistant_text_parts: list[str] = []
        assistant_content = _serialize_assistant_content(list(response.content))
        tool_result_blocks: list[dict[str, Any]] = []
        dispatched_tool_ids: set[str] = set()

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
                dispatched_tool_ids.add(block.id)
                if block.name in WRITE_TOOLS:
                    write_outcomes.append(
                        _WriteToolOutcome(
                            tool_name=block.name,
                            tool_input=tool_input,
                            result=result,
                            is_error=False,
                        )
                    )
                    logger.warning("INSPECT dispatch %s → %s", block.name, result)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            except Exception as exc:
                logger.exception("Error ejecutando tool de Atlas Vital: %s", block.name)
                dispatched_tool_ids.add(block.id)
                if block.name in WRITE_TOOLS:
                    write_outcomes.append(
                        _WriteToolOutcome(
                            tool_name=block.name,
                            tool_input=tool_input,
                            result=None,
                            is_error=True,
                            error_message=_exception_message(exc),
                        )
                    )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error al ejecutar {block.name}: {exc}",
                        "is_error": True,
                    }
                )

        write_outcomes.extend(
            _collect_mcp_write_outcomes(
                list(response.content),
                skip_tool_use_ids=dispatched_tool_ids,
            )
        )

        conversation_messages.append({"role": "assistant", "content": assistant_content})

        if tool_result_blocks:
            conversation_messages.append({"role": "user", "content": tool_result_blocks})
            continue

        assistant_text = "".join(assistant_text_parts).strip()
        return _finalize_user_message(assistant_text, write_outcomes), tools_were_used

    return (
        "Alcanzado el límite de pasadas de herramientas sin respuesta final.",
        tools_were_used,
    )
