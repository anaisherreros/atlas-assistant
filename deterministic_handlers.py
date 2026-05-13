from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from atlas_client import (
    complete_task,
    create_task,
    create_transaction,
    get_calendar,
    get_finance,
    get_tasks_pending,
    get_today,
    log_habit,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeterministicResult:
    text: str
    persist_conversation: bool = False


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(value: str) -> str:
    value = _strip_accents(value.lower())
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.;:-")


def _walk_nodes(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nodes(child)


def _find_first_list(root: Any, aliases: set[str]) -> list[Any]:
    for node in _walk_nodes(root):
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if _normalize_text(str(key)) in aliases and isinstance(value, list):
                return value
    return []


def _find_first_numeric(root: Any, aliases: set[str]) -> float | None:
    for node in _walk_nodes(root):
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if _normalize_text(str(key)) not in aliases:
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                cleaned = value.replace(".", "").replace(",", ".").strip()
                try:
                    return float(cleaned)
                except ValueError:
                    continue
    return None


def _extract_items(
    items: list[Any],
    *,
    id_keys: tuple[str, ...],
    title_keys: tuple[str, ...] = ("title", "name"),
    extra_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        normalized_map = {_normalize_text(str(key)): value for key, value in item.items()}
        title = None
        for key in title_keys:
            if key in normalized_map and normalized_map[key]:
                title = str(normalized_map[key]).strip()
                break
        if not title:
            continue

        identifier = None
        for key in id_keys:
            if key in normalized_map and normalized_map[key] is not None:
                identifier = normalized_map[key]
                break

        dedupe_key = (identifier, _normalize_text(title))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        entry: dict[str, Any] = {"id": identifier, "title": title}
        for key in extra_keys:
            if key in normalized_map:
                entry[key] = normalized_map[key]
        extracted.append(entry)

    return extracted


def _extract_tasks_from_payload(payload: Any) -> list[dict[str, Any]]:
    tasks = _find_first_list(
        payload,
        {
            "tasks",
            "tasks_today",
            "pending_tasks",
            "tareas",
            "today_tasks",
            "items",
            "results",
        },
    )
    return _extract_items(
        tasks,
        id_keys=("task_id", "id"),
        extra_keys=("due_date", "start_time", "end_time", "completed", "status"),
    )


def _extract_habits_from_payload(payload: Any) -> list[dict[str, Any]]:
    habits = _find_first_list(
        payload,
        {"habits", "habits_today", "daily_habits", "today_habits", "items"},
    )
    return _extract_items(
        habits,
        id_keys=("habit_id", "id"),
        extra_keys=("completed", "status", "date"),
    )


def _extract_category_rows(payload: Any) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    seen: set[str] = set()
    candidate_lists = [
        _find_first_list(payload, {"categories", "category_breakdown", "breakdown", "by_category"}),
        _find_first_list(payload, {"gastos_por_categoria", "expenses_by_category"}),
    ]
    for candidate_list in candidate_lists:
        for item in candidate_list:
            if not isinstance(item, dict):
                continue
            normalized_map = {_normalize_text(str(key)): value for key, value in item.items()}
            name = None
            for key in ("category", "name", "title", "label"):
                if key in normalized_map and normalized_map[key]:
                    name = str(normalized_map[key]).strip()
                    break
            if not name:
                continue
            amount = None
            for key in ("amount", "total", "spent", "expense", "value"):
                if key in normalized_map:
                    raw = normalized_map[key]
                    if isinstance(raw, (int, float)):
                        amount = float(raw)
                        break
                    if isinstance(raw, str):
                        try:
                            amount = float(raw.replace(".", "").replace(",", "."))
                            break
                        except ValueError:
                            continue
            if amount is None:
                continue
            normalized_name = _normalize_text(name)
            if normalized_name in seen:
                continue
            seen.add(normalized_name)
            rows.append((name, amount))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:5]


def _format_amount(amount: float | None) -> str | None:
    if amount is None:
        return None
    return f"{amount:,.2f} EUR".replace(",", "_").replace(".", ",").replace("_", ".")


def _format_task_line(item: dict[str, Any]) -> str:
    identifier = f"[{item['id']}] " if item.get("id") is not None else ""
    due_date = item.get("due_date")
    start_time = item.get("start_time")
    end_time = item.get("end_time")
    suffix_parts: list[str] = []
    if due_date:
        suffix_parts.append(str(due_date))
    if start_time and end_time:
        suffix_parts.append(f"{start_time}-{end_time}")
    elif start_time:
        suffix_parts.append(str(start_time))
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"- {identifier}{item['title']}{suffix}"


def _format_habit_line(item: dict[str, Any]) -> str:
    identifier = f"[{item['id']}] " if item.get("id") is not None else ""
    status = item.get("status")
    completed = item.get("completed")
    marker = "x" if completed in (True, 1, "true", "True") or status == "completed" else " "
    return f"- [{marker}] {identifier}{item['title']}"


def _json_excerpt(payload: Any, *, max_len: int = 450) -> str:
    text = str(payload)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _today() -> date:
    return datetime.now().date()


def _extract_date_token(text: str) -> tuple[str | None, tuple[int, int] | None]:
    lowered = text.lower()
    relative_patterns = (
        (r"\bpasado\s+mañana\b|\bpasado\s+manana\b", 2),
        (r"\bmañana\b|\bmanana\b", 1),
        (r"\bhoy\b", 0),
        (r"\bayer\b", -1),
    )
    for pattern, delta_days in relative_patterns:
        match = re.search(pattern, lowered)
        if match:
            parsed = _today() + timedelta(days=delta_days)
            return parsed.isoformat(), match.span()

    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        try:
            parsed = datetime.strptime(iso_match.group(0), "%Y-%m-%d").date()
            return parsed.isoformat(), iso_match.span()
        except ValueError:
            return None, None

    slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
    if slash_match:
        day = int(slash_match.group(1))
        month = int(slash_match.group(2))
        year_group = slash_match.group(3)
        year = _today().year if year_group is None else int(year_group)
        if year < 100:
            year += 2000
        try:
            parsed = date(year=year, month=month, day=day)
            return parsed.isoformat(), slash_match.span()
        except ValueError:
            return None, None

    return None, None


def _extract_date_range(text: str) -> tuple[str | None, str | None]:
    dates: list[str] = []
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        try:
            dates.append(datetime.strptime(match.group(0), "%Y-%m-%d").date().isoformat())
        except ValueError:
            continue
    if len(dates) >= 2:
        return dates[0], dates[1]

    for match in re.finditer(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text):
        parsed, _ = _extract_date_token(match.group(0))
        if parsed:
            dates.append(parsed)
    if len(dates) >= 2:
        return dates[0], dates[1]

    normalized = _normalize_text(text)
    if "hoy" in normalized and "manana" in normalized:
        return _today().isoformat(), (_today() + timedelta(days=1)).isoformat()

    return None, None


def _remove_span(text: str, span: tuple[int, int] | None, *, trim_preposition: bool = False) -> str:
    if span is None:
        return text
    start, end = span
    prefix = text[:start]
    suffix = text[end:]
    if trim_preposition:
        prefix = re.sub(r"(?:\bpara|\bel|\bde)\s*$", "", prefix, flags=re.IGNORECASE)
    return _collapse_spaces(f"{prefix} {suffix}")


def _extract_priority(text: str) -> tuple[str | None, tuple[int, int] | None]:
    priority_map = {"alta": "high", "media": "medium", "baja": "low"}
    match = re.search(r"\bprioridad\s+(alta|media|baja)\b", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return priority_map[match.group(1).lower()], match.span()


def _extract_amount(text: str) -> tuple[float | None, tuple[int, int] | None]:
    match = re.search(r"(-?\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?)?\b", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    raw = match.group(1).replace(",", ".")
    try:
        return float(raw), match.span()
    except ValueError:
        return None, None


def _match_create_task(normalized: str) -> bool:
    return any(
        normalized.startswith(prefix)
        for prefix in (
            "crea una tarea",
            "crea tarea",
            "nueva tarea",
            "apunta tarea",
        )
    )


def _match_complete_task(normalized: str) -> bool:
    return any(
        normalized.startswith(prefix)
        for prefix in (
            "completa tarea",
            "marca tarea",
            "check tarea",
        )
    )


def _match_mark_habit(normalized: str) -> bool:
    return any(
        normalized.startswith(prefix)
        for prefix in (
            "marca habito",
            "marca hábito",
            "marca el habito",
            "marca el hábito",
            "completa habito",
            "completa hábito",
        )
    )


def _match_today_query(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "que tengo hoy",
            "qué tengo hoy",
            "que tengo para hoy",
            "qué tengo para hoy",
            "mi dia",
            "mi día",
        )
    )


def _match_pending_tasks_query(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "mis tareas",
            "tareas pendientes",
            "tareas que tengo pendientes",
        )
    )


def _match_calendar_query(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "calendario",
            "agenda entre",
            "entre fechas",
            "rango de fechas",
        )
    )


def _match_finance_query(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "gastos del mes",
            "cuanto he gastado",
            "cuánto he gastado",
            "cuanto llevo gastado",
            "cuánto llevo gastado",
            "mis finanzas",
            "balance del mes",
            "ingresos del mes",
        )
    )


def _match_create_transaction(normalized: str) -> bool:
    starts = (
        "registra gasto",
        "registra ingreso",
        "apunta gasto",
        "apunta ingreso",
        "anota gasto",
        "anota ingreso",
    )
    return any(normalized.startswith(prefix) for prefix in starts)


def _strip_command_prefix(text: str, patterns: tuple[str, ...]) -> str:
    for pattern in patterns:
        stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
        if stripped != text.strip():
            return stripped
    return text.strip()


def _resolve_item_by_reference(
    query: str,
    items: list[dict[str, Any]],
    *,
    noun: str,
) -> tuple[dict[str, Any] | None, str | None]:
    id_match = re.search(r"(?:#|id\s+|%s\s+)(\d+)\b" % noun, query, flags=re.IGNORECASE)
    if id_match:
        wanted_id = id_match.group(1)
        for item in items:
            if item.get("id") is not None and str(item["id"]) == wanted_id:
                return item, None
        return None, f"No encontré ningún {noun} con ID {wanted_id}."

    cleaned = re.sub(
        r"^(?:completa|marca|check)\s+(?:la\s+|el\s+)?%s\s+" % noun,
        "",
        query,
        flags=re.IGNORECASE,
    )
    cleaned = _collapse_spaces(cleaned)
    if not cleaned:
        return None, f"Necesito que me digas qué {noun} quieres marcar."

    normalized_query = _normalize_text(cleaned)
    exact = [item for item in items if _normalize_text(item["title"]) == normalized_query]
    if len(exact) == 1:
        return exact[0], None
    partial = [
        item
        for item in items
        if normalized_query in _normalize_text(item["title"])
        or _normalize_text(item["title"]) in normalized_query
    ]
    if len(partial) == 1:
        return partial[0], None
    if len(partial) > 1:
        options = "\n".join(f"- [{item.get('id', '?')}] {item['title']}" for item in partial[:5])
        return None, f"He encontrado varios {noun}s parecidos. Dime el ID exacto:\n{options}"
    return None, f"No encontré ningún {noun} que encaje con \"{cleaned}\"."


def _parse_create_task(text: str) -> tuple[dict[str, Any] | None, str | None]:
    remainder = _strip_command_prefix(
        text,
        (
            r"^crea\s+una\s+tarea\s+",
            r"^crea\s+tarea\s+",
            r"^nueva\s+tarea\s+",
            r"^apunta\s+tarea\s+",
        ),
    )
    due_date, date_span = _extract_date_token(remainder)
    priority, priority_span = _extract_priority(remainder)

    title = remainder
    title = _remove_span(title, date_span, trim_preposition=True)
    title = _remove_span(title, priority_span)
    title = _collapse_spaces(title)

    if not due_date:
        return None, "Para crear la tarea necesito una fecha clara, por ejemplo hoy, mañana o 2026-05-20."
    if not title:
        return None, "Para crear la tarea necesito también un título."

    payload: dict[str, Any] = {"title": title, "due_date": due_date}
    if priority:
        payload["priority"] = priority
    return payload, None


def _parse_create_transaction(text: str) -> tuple[dict[str, Any] | None, str | None]:
    normalized = _normalize_text(text)
    transaction_type = "income" if " ingreso" in f" {normalized}" else "expense"
    remainder = _strip_command_prefix(
        text,
        (
            r"^(?:registra|apunta|anota)\s+gasto\s+",
            r"^(?:registra|apunta|anota)\s+ingreso\s+",
        ),
    )
    amount, amount_span = _extract_amount(remainder)
    if amount is None:
        return None, "Para registrar el movimiento necesito el importe."

    tx_date, date_span = _extract_date_token(remainder)
    if tx_date is None:
        tx_date = _today().isoformat()

    description = remainder
    description = _remove_span(description, amount_span)
    description = _remove_span(description, date_span, trim_preposition=True)
    description = _collapse_spaces(description)
    description = re.sub(r"^(?:de|en)\s+", "", description, flags=re.IGNORECASE)
    if not description:
        return None, "Para registrar el movimiento necesito una descripción, por ejemplo gasolina o supermercado."

    return {
        "description": description,
        "amount": amount,
        "transaction_type": transaction_type,
        "date": tx_date,
    }, None


async def _handle_today_query() -> DeterministicResult | None:
    payload = await get_today()
    tasks = _extract_tasks_from_payload(payload)
    habits = _extract_habits_from_payload(payload)
    if not tasks and not habits:
        return DeterministicResult(
            f"Ya consulté Atlas para hoy, pero no pude resumirlo bien de forma determinista:\n{_json_excerpt(payload)}"
        )

    lines = ["Esto es lo que tienes hoy:"]
    lines.append("")
    lines.append("Tareas:")
    if tasks:
        lines.extend(_format_task_line(item) for item in tasks[:12])
    else:
        lines.append("- No veo tareas para hoy.")

    lines.append("")
    lines.append("Hábitos:")
    if habits:
        lines.extend(_format_habit_line(item) for item in habits[:12])
    else:
        lines.append("- No veo hábitos pendientes para hoy.")

    return DeterministicResult("\n".join(lines))


async def _handle_pending_tasks_query() -> DeterministicResult | None:
    payload = await get_tasks_pending()
    tasks = _extract_tasks_from_payload(payload)
    if not tasks:
        return DeterministicResult(
            f"No encontré tareas pendientes o no pude resumirlas con seguridad.\n{_json_excerpt(payload)}"
        )

    lines = [f"Tienes {len(tasks)} tareas pendientes:"]
    lines.extend(_format_task_line(item) for item in tasks[:20])
    return DeterministicResult("\n".join(lines))


async def _handle_calendar_query(text: str) -> DeterministicResult | None:
    start_date, end_date = _extract_date_range(text)
    if not start_date or not end_date:
        return DeterministicResult(
            "Para consultar el calendario necesito dos fechas claras, por ejemplo 2026-05-13 y 2026-05-20."
        )

    payload = await get_calendar(start_date, end_date)
    tasks = _extract_tasks_from_payload(payload)
    habits = _extract_habits_from_payload(payload)
    if not tasks and not habits:
        return DeterministicResult(
            f"Calendario entre {start_date} y {end_date}:\n{_json_excerpt(payload)}"
        )

    lines = [f"Calendario entre {start_date} y {end_date}:"]
    if tasks:
        lines.append("")
        lines.append("Tareas:")
        lines.extend(_format_task_line(item) for item in tasks[:20])
    if habits:
        lines.append("")
        lines.append("Hábitos:")
        lines.extend(_format_habit_line(item) for item in habits[:20])
    return DeterministicResult("\n".join(lines))


async def _handle_finance_query() -> DeterministicResult | None:
    payload = await get_finance()
    expense_total = _find_first_numeric(
        payload,
        {"expense_total", "expenses_total", "total_expenses", "spent_total", "gastos_total", "total_gastos"},
    )
    income_total = _find_first_numeric(
        payload,
        {"income_total", "incomes_total", "total_income", "ingresos_total", "total_ingresos"},
    )
    balance = _find_first_numeric(payload, {"balance", "net", "net_balance", "saldo"})
    categories = _extract_category_rows(payload)

    lines = ["Resumen financiero del mes actual:"]
    if expense_total is not None:
        lines.append(f"- Gastos: {_format_amount(expense_total)}")
    if income_total is not None:
        lines.append(f"- Ingresos: {_format_amount(income_total)}")
    if balance is not None:
        lines.append(f"- Balance: {_format_amount(balance)}")
    if categories:
        lines.append("- Top categorías:")
        lines.extend(f"  - {name}: {_format_amount(amount)}" for name, amount in categories)

    if len(lines) == 1:
        lines.append(_json_excerpt(payload))
    return DeterministicResult("\n".join(lines))


async def _handle_create_task(text: str) -> DeterministicResult | None:
    payload, error = _parse_create_task(text)
    if error:
        return DeterministicResult(error)
    assert payload is not None
    created = await create_task(**payload)
    priority = payload.get("priority", "medium")
    return DeterministicResult(
        "Tarea creada:\n"
        f"- Título: {payload['title']}\n"
        f"- Fecha: {payload['due_date']}\n"
        f"- Prioridad: {priority}\n"
        f"- Respuesta Atlas: {_json_excerpt(created, max_len=220)}"
    )


async def _handle_complete_task(text: str) -> DeterministicResult | None:
    payload = await get_tasks_pending()
    tasks = _extract_tasks_from_payload(payload)
    if not tasks:
        return DeterministicResult("No veo tareas pendientes para poder completar una.")

    task, error = _resolve_item_by_reference(text, tasks, noun="tarea")
    if error:
        return DeterministicResult(error)
    assert task is not None and task.get("id") is not None
    result = await complete_task(int(task["id"]))
    return DeterministicResult(
        f"Tarea completada: [{task['id']}] {task['title']}\n"
        f"Respuesta Atlas: {_json_excerpt(result, max_len=220)}"
    )


async def _handle_mark_habit(text: str) -> DeterministicResult | None:
    payload = await get_today()
    habits = _extract_habits_from_payload(payload)
    if not habits:
        return DeterministicResult("No veo hábitos disponibles hoy para marcar.")

    habit, error = _resolve_item_by_reference(text, habits, noun="habito")
    if error:
        return DeterministicResult(error)
    assert habit is not None and habit.get("id") is not None

    completion_date, _ = _extract_date_token(text)
    if completion_date is None:
        completion_date = _today().isoformat()

    result = await log_habit(
        habit_id=int(habit["id"]),
        date=completion_date,
        completed=True,
    )
    return DeterministicResult(
        f"Hábito marcado: [{habit['id']}] {habit['title']} ({completion_date})\n"
        f"Respuesta Atlas: {_json_excerpt(result, max_len=220)}"
    )


async def _handle_create_transaction(text: str) -> DeterministicResult | None:
    payload, error = _parse_create_transaction(text)
    if error:
        return DeterministicResult(error)
    assert payload is not None
    result = await create_transaction(**payload)
    transaction_label = "Ingreso" if payload["transaction_type"] == "income" else "Gasto"
    return DeterministicResult(
        f"{transaction_label} registrado:\n"
        f"- Descripción: {payload['description']}\n"
        f"- Importe: {_format_amount(payload['amount'])}\n"
        f"- Fecha: {payload['date']}\n"
        f"- Respuesta Atlas: {_json_excerpt(result, max_len=220)}"
    )


async def try_handle_deterministic_message(text: str) -> DeterministicResult | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None

    if _match_create_task(normalized):
        logger.info("Ruta determinista: create_task")
        return await _handle_create_task(text)
    if _match_complete_task(normalized):
        logger.info("Ruta determinista: complete_task")
        return await _handle_complete_task(text)
    if _match_mark_habit(normalized):
        logger.info("Ruta determinista: log_habit_completion")
        return await _handle_mark_habit(text)
    if _match_create_transaction(normalized):
        logger.info("Ruta determinista: create_transaction")
        return await _handle_create_transaction(text)
    if _match_today_query(normalized):
        logger.info("Ruta determinista: get_today")
        return await _handle_today_query()
    if _match_pending_tasks_query(normalized):
        logger.info("Ruta determinista: get_tasks_pending")
        return await _handle_pending_tasks_query()
    if _match_calendar_query(normalized):
        logger.info("Ruta determinista: get_calendar")
        return await _handle_calendar_query(text)
    if _match_finance_query(normalized):
        logger.info("Ruta determinista: get_finance")
        return await _handle_finance_query()
    return None
