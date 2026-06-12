from __future__ import annotations

import re
import unicodedata
from typing import Any

from atlas_client import get_finance

_CATEGORY_LIST_ALIASES = {"categories", "by_category", "category_breakdown"}


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return re.sub(r"\s+", " ", "".join(ch for ch in normalized if not unicodedata.combining(ch))).strip()


def extract_finance_categories(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    categories = payload.get("categories")
    if isinstance(categories, list):
        return [row for row in categories if isinstance(row, dict) and row.get("id") is not None]

    summary = payload.get("summary")
    if isinstance(summary, dict):
        by_category = summary.get("by_category")
        if isinstance(by_category, list):
            return [
                {
                    "id": row.get("category_id") or row.get("id"),
                    "name": row.get("name"),
                    "parent_id": row.get("parent_id"),
                    "category_type": row.get("category_type"),
                }
                for row in by_category
                if isinstance(row, dict) and (row.get("category_id") or row.get("id")) is not None
            ]

    for key in _CATEGORY_LIST_ALIASES:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [row for row in candidate if isinstance(row, dict) and row.get("id") is not None]
    return []


def _category_label(category: dict[str, Any]) -> str:
    name = str(category.get("name") or "").strip()
    parent_id = category.get("parent_id")
    if parent_id:
        return name
    return name


def match_finance_category(
    categories: list[dict[str, Any]],
    query: str,
    *,
    transaction_type: str = "expense",
) -> dict[str, Any] | None:
    normalized_query = _normalize_label(query)
    if not normalized_query or not categories:
        return None

    typed = [
        category
        for category in categories
        if _category_matches_transaction_type(category, transaction_type)
    ]
    pool = typed or categories

    exact = [
        category
        for category in pool
        if _normalize_label(str(category.get("name") or "")) == normalized_query
    ]
    if len(exact) == 1:
        return exact[0]

    partial = [
        category
        for category in pool
        if normalized_query in _normalize_label(str(category.get("name") or ""))
        or _normalize_label(str(category.get("name") or "")) in normalized_query
    ]
    if len(partial) == 1:
        return partial[0]

    alias_matches = [
        category
        for category in pool
        if normalized_query in _category_aliases(category)
    ]
    if len(alias_matches) == 1:
        return alias_matches[0]
    return None


def _category_matches_transaction_type(category: dict[str, Any], transaction_type: str) -> bool:
    category_type = str(category.get("category_type") or "").lower()
    if transaction_type == "income":
        return category_type in {"income", "investment", "savings", "tithe", "donation", ""}
    return category_type in {"basic", "entertainment", "investment", "tithe", "donation", "savings", ""}


def _category_aliases(category: dict[str, Any]) -> set[str]:
    name = _normalize_label(str(category.get("name") or ""))
    aliases = {name}
    category_type = str(category.get("category_type") or "").lower()
    defaults = {
        "basic": {"comida", "supermercado", "alimentacion", "alimentación", "comestibles", "mercadona", "lidl", "coop"},
        "entertainment": {"ocio", "restaurante", "cafe", "café", "bar", "cine", "salir"},
        "income": {"salario", "nomina", "nómina", "ingreso", "cobro"},
    }
    aliases.update(defaults.get(category_type, set()))
    if "comida" in name or "aliment" in name:
        aliases.update({"comida", "supermercado", "lidl", "mercadona"})
    return aliases


def format_category_options(categories: list[dict[str, Any]], *, limit: int = 8) -> str:
    lines: list[str] = []
    for category in categories[:limit]:
        label = _category_label(category)
        parent_id = category.get("parent_id")
        suffix = " (sub)" if parent_id else ""
        lines.append(f"[{category.get('id')}] {label}{suffix}")
    return ", ".join(lines)


async def resolve_finance_category_id(
    *,
    category_name: str | None = None,
    category_id: int | None = None,
    transaction_type: str = "expense",
) -> tuple[int | None, str | None]:
    if category_id is not None:
        return int(category_id), None
    if not category_name or not str(category_name).strip():
        return None, None

    payload = await get_finance()
    categories = extract_finance_categories(payload)
    if not categories:
        return None, "No hay categorías financieras configuradas en Atlas Vital."

    matched = match_finance_category(
        categories,
        str(category_name).strip(),
        transaction_type=transaction_type,
    )
    if matched and matched.get("id") is not None:
        return int(matched["id"]), None

    options = format_category_options(categories)
    return None, f"No encontré la categoría '{category_name}'. Disponibles: {options}"
