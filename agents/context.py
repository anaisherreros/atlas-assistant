from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any

from atlas_client import (
    get_all_desires_full,
    get_dashboard,
    get_desire_structure,
    get_exercise_recent,
    get_finance_full,
    get_health_context,
    get_health_emotional,
    get_reviews_summary,
)

logger = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(stripped.split())


def _walk_nodes(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_nodes(child)


def _extract_desire_candidates(root: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for node in _walk_nodes(root):
        if not isinstance(node, dict):
            continue
        desire_id = node.get("desire_id", node.get("id"))
        title = node.get("title") or node.get("name")
        if not isinstance(desire_id, int) or not isinstance(title, str):
            continue
        has_desire_shape = any(
            key in node for key in ("area", "area_id", "status", "priority", "goals", "habits")
        )
        if not has_desire_shape or desire_id in seen_ids:
            continue
        seen_ids.add(desire_id)
        candidates.append(node)

    return candidates


def _collect_text_fields(value: Any) -> list[str]:
    texts: list[str] = []
    for node in _walk_nodes(value):
        if isinstance(node, str):
            texts.append(_normalize_text(node))
    return texts


def _priority_score(value: Any) -> int:
    normalized = _normalize_text(str(value or ""))
    if normalized in {"high", "alta", "urgent", "urgente"}:
        return 3
    if normalized in {"medium", "media"}:
        return 2
    if normalized in {"low", "baja"}:
        return 1
    return 0


def _status_score(value: Any) -> int:
    normalized = _normalize_text(str(value or ""))
    if normalized in {"active", "activo", "in_progress", "en progreso"}:
        return 3
    if normalized in {"planned", "planificado", "pending", "pendiente"}:
        return 2
    if normalized in {"completed", "completado"}:
        return 1
    return 0


def _resolve_area_label(desire: dict[str, Any]) -> str:
    area = desire.get("area")
    if isinstance(area, dict):
        for key in ("title", "name", "slug"):
            value = area.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(area, str) and area.strip():
        return area.strip()

    subarea = desire.get("subarea")
    if isinstance(subarea, dict):
        for key in ("title", "name", "slug"):
            value = subarea.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(subarea, str) and subarea.strip():
        return subarea.strip()

    area_blob = _collect_text_fields(area)
    if area_blob:
        return area_blob[0]
    return "sin_area"


def _summarize_desire(desire: dict[str, Any]) -> dict[str, Any]:
    goals = desire.get("goals")
    habits = desire.get("habits")
    return {
        "id": desire.get("desire_id", desire.get("id")),
        "title": desire.get("title"),
        "status": desire.get("status"),
        "priority": desire.get("priority"),
        "area": _resolve_area_label(desire),
        "goals_count": len(goals) if isinstance(goals, list) else None,
        "habits_count": len(habits) if isinstance(habits, list) else None,
    }


def _sort_desires(desires: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        desires,
        key=lambda desire: (
            _status_score(desire.get("status")),
            _priority_score(desire.get("priority")),
            _normalize_text(str(desire.get("title") or "")),
        ),
        reverse=True,
    )


def _build_desire_overview(
    desires: list[dict[str, Any]],
    *,
    area_keywords: tuple[str, ...] | None = None,
    max_areas: int = 8,
    max_desires_per_area: int = 4,
) -> dict[str, Any]:
    normalized_keywords = (
        tuple(_normalize_text(keyword) for keyword in area_keywords) if area_keywords else ()
    )
    filtered: list[dict[str, Any]] = []

    for desire in desires:
        if normalized_keywords and _desire_match_score(desire, normalized_keywords) <= 0:
            continue
        filtered.append(desire)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for desire in _sort_desires(filtered):
        area_label = _resolve_area_label(desire)
        grouped.setdefault(area_label, []).append(_summarize_desire(desire))

    areas: list[dict[str, Any]] = []
    for area_label, area_desires in grouped.items():
        active_count = sum(
            1 for desire in area_desires if _status_score(desire.get("status")) >= 2
        )
        areas.append(
            {
                "area": area_label,
                "active_count": active_count,
                "desires": area_desires[:max_desires_per_area],
            }
        )

    areas.sort(
        key=lambda area: (
            area["active_count"],
            len(area["desires"]),
            _normalize_text(str(area["area"])),
        ),
        reverse=True,
    )

    return {
        "total_desires": len(filtered),
        "areas": areas[:max_areas],
    }


def _desire_match_score(desire: dict[str, Any], area_keywords: tuple[str, ...]) -> int:
    score = 0
    normalized_keywords = tuple(_normalize_text(keyword) for keyword in area_keywords)
    title = _normalize_text(str(desire.get("title") or desire.get("name") or ""))
    status = _normalize_text(str(desire.get("status") or ""))

    if title:
        for keyword in normalized_keywords:
            if keyword == title:
                score += 100
            elif keyword in title:
                score += 40

    area_blob = " ".join(_collect_text_fields(desire.get("area")))
    if area_blob:
        for keyword in normalized_keywords:
            if keyword in area_blob:
                score += 120

    subarea_blob = " ".join(_collect_text_fields(desire.get("subarea")))
    if subarea_blob:
        for keyword in normalized_keywords:
            if keyword in subarea_blob:
                score += 60

    if status in {"active", "activo", "in_progress", "en progreso"}:
        score += 15

    return score


async def _get_focus_desire_payload(
    *,
    candidates: list[dict[str, Any]],
    area_keywords: tuple[str, ...],
) -> dict[str, Any] | None:
    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=lambda desire: _desire_match_score(desire, area_keywords),
        reverse=True,
    )
    top_match = ranked[0]
    top_score = _desire_match_score(top_match, area_keywords)
    if top_score <= 0:
        return None

    desire_id = top_match.get("desire_id", top_match.get("id"))
    if not isinstance(desire_id, int):
        return None

    structure = await get_desire_structure(desire_id)
    return {
        "match_score": top_score,
        "matched_keywords": list(area_keywords),
        "matched_desire_summary": {
            "id": desire_id,
            "title": top_match.get("title"),
            "status": top_match.get("status"),
            "area": top_match.get("area"),
            "priority": top_match.get("priority"),
        },
        "desire_structure": structure,
    }


async def _load_desire_catalog() -> tuple[Any, list[dict[str, Any]]]:
    all_desires = await get_all_desires_full()
    return all_desires, _extract_desire_candidates(all_desires)


async def fetch_context_for_agent(agent_key: str) -> str:
    """Carga JSON de contexto Atlas según el agente activo."""
    normalized_agent_key = {
        "nutritionist": "performance",
        "trainer": "performance",
    }.get(agent_key, agent_key)
    payload: Any
    try:
        if normalized_agent_key == "personal":
            dashboard = await get_dashboard()
            desire_overview: Any = None
            try:
                _, desire_candidates = await _load_desire_catalog()
                desire_overview = _build_desire_overview(desire_candidates)
            except Exception:
                logger.exception("Fallo construyendo desire_overview para personal")
            payload = {
                "dashboard": dashboard,
                "desire_overview": desire_overview,
            }
        elif normalized_agent_key == "coach":
            dashboard = await get_dashboard()
            reviews_summary = await get_reviews_summary()
            desire_overview: Any = None
            growth_desires: Any = None
            try:
                _, desire_candidates = await _load_desire_catalog()
                desire_overview = _build_desire_overview(desire_candidates)
                growth_desires = _build_desire_overview(
                    desire_candidates,
                    area_keywords=(
                        "proposito",
                        "proyecto",
                        "trabajo",
                        "marca personal",
                        "relaciones",
                        "desarrollo personal",
                        "vida",
                    ),
                    max_areas=5,
                    max_desires_per_area=3,
                )
            except Exception:
                logger.exception("Fallo construyendo contexto de deseos para coach")
            payload = {
                "dashboard": dashboard,
                "reviews_summary": reviews_summary,
                "desire_overview": desire_overview,
                "growth_desires": growth_desires,
            }
        elif normalized_agent_key == "performance":
            health_today: Any = {}
            health_emotional: Any = None
            exercise_recent: Any = {}
            focus_desire: Any = None
            area_desire_overview: Any = None
            try:
                health_today = await get_health_context()
            except Exception:
                logger.exception("Fallo get_health_context (health/today)")
            try:
                health_emotional = await get_health_emotional()
            except Exception as exc:
                logger.warning(
                    "get_health_emotional no disponible o error: %s",
                    exc,
                )
                health_emotional = None
            try:
                exercise_recent = await get_exercise_recent(days=7)
            except Exception:
                logger.exception("Fallo get_exercise_recent")
            try:
                _, desire_candidates = await _load_desire_catalog()
                area_desire_overview = _build_desire_overview(
                    desire_candidates,
                    area_keywords=("salud", "cuerpo", "bienestar", "energia", "fuerza"),
                    max_areas=4,
                    max_desires_per_area=3,
                )
                focus_desire = await _get_focus_desire_payload(
                    candidates=desire_candidates,
                    area_keywords=("salud", "cuerpo", "bienestar", "energia", "fuerza"),
                )
            except Exception:
                logger.exception("Fallo resolviendo deseo foco para performance")
            payload = {
                "health_today": health_today,
                "health_emotional": health_emotional,
                "exercise_recent": exercise_recent,
                "area_desire_overview": area_desire_overview,
                "focus_desire": focus_desire,
            }
        elif normalized_agent_key == "financial":
            finance_full: Any = {}
            focus_desire: Any = None
            area_desire_overview: Any = None
            try:
                finance_full = await get_finance_full()
            except Exception:
                logger.exception("Fallo get_finance_full")
            try:
                _, desire_candidates = await _load_desire_catalog()
                area_desire_overview = _build_desire_overview(
                    desire_candidates,
                    area_keywords=(
                        "finanzas",
                        "dinero",
                        "abundancia",
                        "patrimonio",
                        "libertad financiera",
                    ),
                    max_areas=4,
                    max_desires_per_area=3,
                )
                focus_desire = await _get_focus_desire_payload(
                    candidates=desire_candidates,
                    area_keywords=("finanzas", "dinero", "abundancia", "patrimonio", "libertad financiera"),
                )
            except Exception:
                logger.exception("Fallo resolviendo deseo foco para financial")
            payload = {
                "finance_full": finance_full,
                "area_desire_overview": area_desire_overview,
                "focus_desire": focus_desire,
            }
        else:
            payload = await get_dashboard()
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        logger.exception("Error cargando contexto Atlas para agente %s", normalized_agent_key)
        return "{}"
