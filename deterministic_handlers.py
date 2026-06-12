from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from atlas_client import (
    complete_task,
    create_habit,
    create_task,
    create_transaction,
    get_calendar,
    get_finance,
    get_tasks_pending,
    get_today,
    log_habit,
    update_health,
)
from finance_categories import resolve_finance_category_id
from health_helpers import extract_weight_kg_from_text

logger = logging.getLogger(__name__)

_DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "CHF")
_ZURICH_TZ = ZoneInfo("Europe/Zurich")


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
        extra_keys=("completed", "status", "date", "start_time", "end_time"),
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
    formatted = f"{amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{formatted} {_DEFAULT_CURRENCY}"


def _format_time_window(start_time: object, end_time: object) -> str:
    start_value = str(start_time or "").strip()
    end_value = str(end_time or "").strip()
    if start_value and end_value:
        return f"{start_value}-{end_value}"
    if start_value:
        return start_value
    return "-"


def _format_task_line(item: dict[str, Any]) -> str:
    identifier = f"[{item['id']}] " if item.get("id") is not None else ""
    due_date = item.get("due_date")
    suffix_parts: list[str] = []
    if due_date:
        suffix_parts.append(str(due_date))
    time_window = _format_time_window(item.get("start_time"), item.get("end_time"))
    if time_window != "-":
        suffix_parts.append(time_window)
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return f"- {identifier}{item['title']}{suffix}"


def _format_habit_line(item: dict[str, Any]) -> str:
    identifier = f"[{item['id']}] " if item.get("id") is not None else ""
    status = item.get("status")
    completed = item.get("completed")
    marker = "x" if completed in (True, 1, "true", "True") or status == "completed" else " "
    time_window = _format_time_window(item.get("start_time"), item.get("end_time"))
    if time_window != "-":
        return f"- [{marker}] {time_window} · {identifier}{item['title']}"
    return f"- [{marker}] {identifier}{item['title']}"


def _json_excerpt(payload: Any, *, max_len: int = 450) -> str:
    text = str(payload)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _today() -> date:
    return datetime.now(_ZURICH_TZ).date()


def _api_created_flag(api_response: Any) -> bool | None:
    if isinstance(api_response, dict) and "created" in api_response:
        return bool(api_response["created"])
    return None


def _extract_transaction_category(api_response: Any) -> str | None:
    if not isinstance(api_response, dict):
        return None
    transaction = api_response.get("transaction")
    if not isinstance(transaction, dict):
        return None
    category = transaction.get("category")
    if isinstance(category, dict):
        for key in ("name", "title", "label"):
            value = category.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(category, str) and category.strip():
        return category.strip()
    return None


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


def _normalize_clock(hour_text: str, minute_text: str | None = None) -> str | None:
    hour = int(hour_text)
    minute = 0 if minute_text is None else int(minute_text)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _extract_time_range(text: str) -> tuple[str | None, str | None, tuple[int, int] | None]:
    patterns = (
        r"\b(?:de|desde)\s+(?:las?\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:h)?\s+(?:a|hasta)\s+(?:las?\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:h)?\b",
        r"\ba\s+las?\s+(\d{1,2})(?::(\d{2}))?\s*(?:h)?\s+(?:a|hasta)\s+las?\s+(\d{1,2})(?::(\d{2}))?\s*(?:h)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        start_time = _normalize_clock(match.group(1), match.group(2))
        end_time = _normalize_clock(match.group(3), match.group(4))
        if start_time and end_time:
            return start_time, end_time, match.span()
    return None, None, None


def _extract_time_token(text: str) -> tuple[str | None, tuple[int, int] | None]:
    patterns = (
        r"\ba\s+las?\s+(\d{1,2})(?::(\d{2}))?\s*(?:h)?\b",
        r"\bpara\s+las?\s+(\d{1,2})(?::(\d{2}))?\s*(?:h)?\b",
        r"\bsobre\s+las?\s+(\d{1,2})(?::(\d{2}))?\s*(?:h)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = _normalize_clock(match.group(1), match.group(2))
        if parsed:
            return parsed, match.span()
    return None, None


def _extract_time_info(
    text: str,
) -> tuple[str | None, str | None, tuple[int, int] | None]:
    start_time, end_time, range_span = _extract_time_range(text)
    if start_time:
        return start_time, end_time, range_span
    start_time, time_span = _extract_time_token(text)
    return start_time, None, time_span


def _remove_span(text: str, span: tuple[int, int] | None, *, trim_preposition: bool = False) -> str:
    if span is None:
        return text
    start, end = span
    prefix = text[:start]
    suffix = text[end:]
    if trim_preposition:
        prefix = re.sub(r"(?:\bpara|\bel|\bde)\s*$", "", prefix, flags=re.IGNORECASE)
    return _collapse_spaces(f"{prefix} {suffix}")


def _remove_spans(text: str, span_specs: list[tuple[tuple[int, int] | None, bool]]) -> str:
    cleaned = text
    sortable_specs = [spec for spec in span_specs if spec[0] is not None]
    for span, trim_preposition in sorted(
        sortable_specs,
        key=lambda spec: spec[0][0],  # type: ignore[index]
        reverse=True,
    ):
        cleaned = _remove_span(cleaned, span, trim_preposition=trim_preposition)
    return _collapse_spaces(cleaned)


def _extract_priority(text: str) -> tuple[str | None, tuple[int, int] | None]:
    priority_map = {"alta": "high", "media": "medium", "baja": "low"}
    match = re.search(r"\bprioridad\s+(alta|media|baja)\b", text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return priority_map[match.group(1).lower()], match.span()


def _extract_amount(text: str) -> tuple[float | None, tuple[int, int] | None]:
    match = re.search(
        r"(-?\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|chf|fr\.?|francos?|francs?|s\.?)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    raw = match.group(1).replace(",", ".")
    try:
        return float(raw), match.span()
    except ValueError:
        return None, None


_EXPENSE_VERB_RE = re.compile(
    r"\b(gaste|gast[eé]|pague|pag[eé]|compre|compr[eé]|he\s+gastado|gastado|cobr[eé]|recib[ií]|ingreso)\b",
    re.IGNORECASE,
)
_WEIGHT_INTENT_RE = re.compile(
    r"\b(pesad[oa]|peso|me\s+pese|me\s+he\s+pesado|kilos?|kg)\b",
    re.IGNORECASE,
)
_HABIT_INTENT_RE = re.compile(
    r"\b(hice|marque|marqu[eé]|complete|complet[eé]|ya\s+hice)\b",
    re.IGNORECASE,
)
_CURRENCY_AMOUNT_RE = re.compile(
    r"(-?\d+(?:[.,]\d{1,2})?)\s*(?:€|euros?|chf|fr\.?|francos?|francs?)\b",
    re.IGNORECASE,
)
_CATEGORY_HINT_WORDS = frozenset({
    "comida",
    "ocio",
    "transporte",
    "salud",
    "ropa",
    "supermercado",
    "restaurante",
    "gasolina",
    "farmacia",
    "casa",
    "ahorro",
})


def _is_weight_number_match(match: re.Match[str], text: str) -> bool:
    tail = text[match.end() : match.end() + 12].lower()
    return bool(re.match(r"\s*(?:kg|kilos?|kgs)\b", tail))


def _operational_intent_keys(normalized: str) -> set[str]:
    intents: set[str] = set()
    if _EXPENSE_VERB_RE.search(normalized):
        intents.add("expense")
    if _WEIGHT_INTENT_RE.search(normalized):
        intents.add("weight")
    if _HABIT_INTENT_RE.search(normalized):
        intents.add("habit")
    return intents


def _is_compound_operational_message(normalized: str) -> bool:
    return len(_operational_intent_keys(normalized)) > 1


def _extract_expense_clause(text: str) -> str | None:
    clauses = re.split(r"\s+y\s+", text.strip(), flags=re.IGNORECASE)
    expense_clauses = [clause.strip() for clause in clauses if _EXPENSE_VERB_RE.search(clause)]
    if len(expense_clauses) == 1:
        return expense_clauses[0]
    if len(expense_clauses) > 1:
        return None
    if _EXPENSE_VERB_RE.search(text):
        return text.strip()
    return None


def _extract_expense_amount(text: str) -> tuple[float | None, tuple[int, int] | None]:
    for match in _CURRENCY_AMOUNT_RE.finditer(text):
        if _is_weight_number_match(match, text):
            continue
        try:
            return float(match.group(1).replace(",", ".")), match.span()
        except ValueError:
            continue

    verb_match = _EXPENSE_VERB_RE.search(text)
    if verb_match:
        after_verb = text[verb_match.end() : verb_match.end() + 24]
        amount_match = re.search(r"(-?\d+(?:[.,]\d{1,2})?)", after_verb)
        if amount_match and not _is_weight_number_match(amount_match, after_verb):
            start = verb_match.end() + amount_match.start()
            end = verb_match.end() + amount_match.end()
            try:
                return float(amount_match.group(1).replace(",", ".")), (start, end)
            except ValueError:
                pass

    for match in re.finditer(r"(-?\d+(?:[.,]\d{1,2})?)", text):
        if _is_weight_number_match(match, text):
            continue
        try:
            return float(match.group(1).replace(",", ".")), match.span()
        except ValueError:
            continue
    return None, None


def _extract_transaction_category_name(remainder: str) -> str | None:
    de_match = re.search(r"\bde\s+([\w\sáéíóúñü'-]+)\s*$", remainder, flags=re.IGNORECASE)
    if de_match:
        return _collapse_spaces(de_match.group(1))

    en_match = re.search(r"\ben\s+(?:el|la|los|las)?\s*([\w\sáéíóúñü'-]+)\s*$", remainder, flags=re.IGNORECASE)
    if not en_match:
        return None
    candidate = _collapse_spaces(en_match.group(1))
    normalized_candidate = _normalize_text(candidate)
    if normalized_candidate in _CATEGORY_HINT_WORDS or len(candidate.split()) <= 2:
        return candidate
    return None


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


def _match_help_query(normalized: str) -> bool:
    return normalized in {
        "ayuda",
        "help",
        "comandos",
        "ejemplos",
        "que puedes hacer",
        "qué puedes hacer",
        "como te hablo",
        "cómo te hablo",
    }


def _match_complete_task(normalized: str) -> bool:
    return any(
        normalized.startswith(prefix)
        for prefix in (
            "completa tarea",
            "marca tarea",
            "check tarea",
        )
    )


def _match_create_habit(normalized: str) -> bool:
    return any(
        normalized.startswith(prefix)
        for prefix in (
            "crea un habito",
            "crea una rutina",
            "crea habito",
            "crea hábito",
            "nuevo habito",
            "nuevo hábito",
            "apunta habito",
            "apunta hábito",
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


def _match_natural_expense(normalized: str) -> bool:
    if _match_create_transaction(normalized):
        return False
    if _is_compound_operational_message(normalized):
        return False
    if not re.search(r"\d", normalized):
        return False
    return bool(_EXPENSE_VERB_RE.search(normalized))


def _match_log_weight(normalized: str) -> bool:
    if not _WEIGHT_INTENT_RE.search(normalized):
        return False
    if not re.search(r"\d", normalized):
        return False
    if _is_compound_operational_message(normalized):
        return False
    return True


def _parse_log_weight(text: str) -> tuple[dict[str, Any] | None, str | None]:
    weight_kg = extract_weight_kg_from_text(text)
    if weight_kg is None:
        return None, "No pude identificar el peso en kg."

    tx_date, _ = _extract_date_token(text)
    if tx_date is None:
        tx_date = _today().isoformat()

    return {"date": tx_date, "weight_kg": weight_kg}, None


async def _handle_log_weight(text: str) -> DeterministicResult | None:
    payload, error = _parse_log_weight(text)
    if error:
        return DeterministicResult(error)
    if payload is None:
        return None

    result = await update_health(
        date=str(payload["date"]),
        physical={"weight_kg": payload["weight_kg"]},
    )
    physical = result.get("physical_log") if isinstance(result, dict) else None
    saved_weight = payload["weight_kg"]
    if isinstance(physical, dict) and physical.get("weight_kg") is not None:
        saved_weight = physical["weight_kg"]
    return DeterministicResult(
        f"Peso registrado: {saved_weight} kg · {payload['date']}\n"
        f"Respuesta Atlas: {_json_excerpt(result, max_len=220)}"
    )


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


def _extract_frequency(text: str) -> tuple[str | None, tuple[int, int] | None]:
    patterns = (
        ("daily", r"\b(diario|diaria|cada\s+dia|cada\s+día|todos?\s+los\s+dias|todos?\s+los\s+días)\b"),
        ("weekly", r"\b(semanal|cada\s+semana)\b"),
        ("monthly", r"\b(mensual|cada\s+mes)\b"),
    )
    for frequency, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return frequency, match.span()
    return None, None


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
    start_time, end_time, time_span = _extract_time_info(remainder)

    title = _remove_spans(
        remainder,
        [
            (date_span, True),
            (priority_span, False),
            (time_span, True),
        ],
    )

    if due_date is None and start_time:
        due_date = _today().isoformat()
    if not due_date:
        due_date = _today().isoformat()
    if not title:
        return None, "Para crear la tarea necesito también un título."

    payload: dict[str, Any] = {"title": title, "due_date": due_date}
    if priority:
        payload["priority"] = priority
    if start_time:
        payload["start_time"] = start_time
    if end_time:
        payload["end_time"] = end_time
    return payload, None


def _parse_create_habit(text: str) -> tuple[dict[str, Any] | None, str | None]:
    remainder = _strip_command_prefix(
        text,
        (
            r"^crea\s+un\s+habito\s+",
            r"^crea\s+un\s+hábito\s+",
            r"^crea\s+una\s+rutina\s+",
            r"^crea\s+habito\s+",
            r"^crea\s+hábito\s+",
            r"^nuevo\s+habito\s+",
            r"^nuevo\s+hábito\s+",
            r"^apunta\s+habito\s+",
            r"^apunta\s+hábito\s+",
        ),
    )
    start_date, date_span = _extract_date_token(remainder)
    frequency_type, frequency_span = _extract_frequency(remainder)
    start_time, end_time, time_span = _extract_time_info(remainder)

    title = _remove_spans(
        remainder,
        [
            (date_span, True),
            (frequency_span, False),
            (time_span, True),
        ],
    )

    if not title:
        return None, "Para crear el hábito necesito también un título."

    payload: dict[str, Any] = {
        "title": title,
        "start_date": start_date or _today().isoformat(),
        "frequency_type": frequency_type or "daily",
    }
    if start_time:
        payload["start_time"] = start_time
    if end_time:
        payload["end_time"] = end_time
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

    description = _remove_spans(
        remainder,
        [
            (amount_span, False),
            (date_span, True),
        ],
    )
    description = re.sub(r"^(?:de|en)\s+", "", description, flags=re.IGNORECASE)
    if not description:
        return None, "Para registrar el movimiento necesito una descripción, por ejemplo gasolina o supermercado."

    return {
        "description": description,
        "amount": amount,
        "transaction_type": transaction_type,
        "date": tx_date,
    }, None


def _parse_natural_transaction(text: str) -> tuple[dict[str, Any] | None, str | None]:
    normalized = _normalize_text(text)
    if not _EXPENSE_VERB_RE.search(normalized):
        return None, None
    if _is_compound_operational_message(normalized):
        return None, None

    expense_clause = _extract_expense_clause(text)
    if not expense_clause:
        return None, None

    clause_normalized = _normalize_text(expense_clause)
    transaction_type = "expense"
    if re.search(r"\b(ingreso|cobre|cobr[eé]|recib[ií]|me\s+pagaron|me\s+entr[oó])\b", clause_normalized):
        transaction_type = "income"

    remainder = expense_clause.strip()
    tx_date, date_span = _extract_date_token(remainder)
    if tx_date is None:
        tx_date = _today().isoformat()
        date_span = None

    amount, amount_span = _extract_expense_amount(remainder)
    if amount is None:
        return None, "Para registrar el movimiento necesito el importe."

    category_name = _extract_transaction_category_name(remainder)

    working = _remove_spans(
        remainder,
        [
            (date_span, True),
            (amount_span, False),
        ],
    )
    working = _EXPENSE_VERB_RE.sub("", working, count=1).strip()
    if category_name:
        working = re.sub(
            rf"\s+(?:de|en)\s+(?:el|la|los|las)?\s*{re.escape(category_name)}\s*$",
            "",
            working,
            flags=re.IGNORECASE,
        ).strip()

    description = None
    en_match = re.search(
        r"\ben\s+(?:el|la|los|las)?\s*([\w\d\sáéíóúñü'&.-]+?)(?:\s+de\s+[\w\sáéíóúñ'-]+)?\s*$",
        working,
        flags=re.IGNORECASE,
    )
    if en_match:
        candidate = _collapse_spaces(en_match.group(1)).title()
        if _normalize_text(candidate) not in _CATEGORY_HINT_WORDS:
            description = candidate

    if not description:
        cleaned = _collapse_spaces(working)
        if cleaned and _normalize_text(cleaned) not in _CATEGORY_HINT_WORDS:
            description = cleaned

    if not description:
        description = category_name or ("Ingreso" if transaction_type == "income" else "Gasto")

    payload: dict[str, Any] = {
        "description": description,
        "amount": amount,
        "transaction_type": transaction_type,
        "date": tx_date,
    }
    if category_name:
        payload["category_name"] = category_name
    return payload, None


async def _handle_help_query() -> DeterministicResult | None:
    lines = [
        "Puedo ayudarte con estos comandos rápidos:",
        "",
        "Tareas:",
        "- crea tarea llamar al dentista mañana a las 15:30",
        "- crea tarea bloque de foco mañana de 15 a 16 prioridad alta",
        "- completa tarea informe",
        "",
        "Hábitos:",
        "- crea habito caminar diario a las 08:00",
        "- crea habito leer desde las 21 hasta las 22",
        "- marca hábito beber agua",
        "",
        "Dinero:",
        "- registra gasto gasolina 45 hoy",
        "- apunta ingreso nomina 2200",
        "",
        "Consultas:",
        "- qué tengo hoy",
        "- mis tareas",
        "- calendario entre 2026-05-14 y 2026-05-20",
        "- gastos del mes",
        "",
        "Consejo rápido: usa fecha como hoy, mañana o 2026-05-20 y hora como 08:00 o de 15 a 16.",
    ]
    return DeterministicResult("\n".join(lines))


async def _handle_today_query() -> DeterministicResult | None:
    payload = await get_today()
    tasks = _extract_tasks_from_payload(payload)
    habits = _extract_habits_from_payload(payload)
    if not tasks and not habits:
        logger.warning("No pude interpretar get_today: %s", payload)
        return DeterministicResult("No pude interpretar la respuesta de Atlas Vital.")

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
    schedule = _format_time_window(payload.get("start_time"), payload.get("end_time"))
    created_flag = _api_created_flag(created)
    if created_flag is False:
        header = f"⚠️ Ya existía: {payload['title']}"
    else:
        header = "Tarea creada:"
    return DeterministicResult(
        f"{header}\n"
        f"- Título: {payload['title']}\n"
        f"- Fecha: {payload['due_date']}\n"
        f"- Hora: {schedule}\n"
        f"- Prioridad: {priority}\n"
        f"- Respuesta Atlas: {_json_excerpt(created, max_len=220)}"
    )


async def _handle_create_habit(text: str) -> DeterministicResult | None:
    payload, error = _parse_create_habit(text)
    if error:
        return DeterministicResult(error)
    assert payload is not None
    result = await create_habit(**payload)
    schedule = _format_time_window(payload.get("start_time"), payload.get("end_time"))
    created_flag = _api_created_flag(result)
    if created_flag is False:
        header = f"⚠️ Ya existía: {payload['title']}"
    else:
        header = "Hábito creado:"
    return DeterministicResult(
        f"{header}\n"
        f"- Título: {payload['title']}\n"
        f"- Inicio: {payload['start_date']}\n"
        f"- Frecuencia: {payload['frequency_type']}\n"
        f"- Hora: {schedule}\n"
        f"- Respuesta Atlas: {_json_excerpt(result, max_len=220)}"
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
    if payload is None:
        payload, error = _parse_natural_transaction(text)
    if error:
        return DeterministicResult(error)
    if payload is None:
        return None

    category_name = payload.pop("category_name", None)
    category_id, category_error = await resolve_finance_category_id(
        category_name=category_name,
        category_id=payload.get("category_id"),
        transaction_type=str(payload["transaction_type"]),
    )
    if category_error:
        return DeterministicResult(category_error)

    result = await create_transaction(
        description=str(payload["description"]),
        amount=float(payload["amount"]),
        transaction_type=str(payload["transaction_type"]),
        date=str(payload["date"]),
        category_id=category_id,
    )
    transaction_label = "Ingreso" if payload["transaction_type"] == "income" else "Gasto"
    category = _extract_transaction_category(result)
    lines = [
        f"{transaction_label} registrado:",
        f"- Descripción: {payload['description']}",
        f"- Importe: {_format_amount(payload['amount'])}",
        f"- Fecha: {payload['date']}",
    ]
    if category:
        lines.append(f"- Categoría: {category}")
    lines.append(f"- Respuesta Atlas: {_json_excerpt(result, max_len=220)}")
    return DeterministicResult("\n".join(lines))


async def try_handle_deterministic_message(text: str) -> DeterministicResult | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None

    if _match_help_query(normalized):
        logger.info("Ruta determinista: help")
        return await _handle_help_query()
    if _match_create_task(normalized):
        logger.info("Ruta determinista: create_task")
        return await _handle_create_task(text)
    if _match_create_habit(normalized):
        logger.info("Ruta determinista: create_habit")
        return await _handle_create_habit(text)
    if _match_complete_task(normalized):
        logger.info("Ruta determinista: complete_task")
        return await _handle_complete_task(text)
    if _match_mark_habit(normalized):
        logger.info("Ruta determinista: log_habit_completion")
        return await _handle_mark_habit(text)
    if _match_log_weight(normalized):
        logger.info("Ruta determinista: log_weight")
        return await _handle_log_weight(text)
    if _match_create_transaction(normalized) or _match_natural_expense(normalized):
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
