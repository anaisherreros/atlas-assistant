from __future__ import annotations

import re
from typing import Any

_PHYSICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "weight_kg": ("weight_kg", "weight", "peso", "peso_kg", "body_weight"),
    "sleep_hours": ("sleep_hours", "sleep", "sueno", "sueño", "horas_sueno", "horas_de_sueno"),
    "note": ("note", "nota", "notes"),
    "calorie_deficit": ("calorie_deficit", "deficit_calorico", "deficit"),
}

_BODY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "weight_kg": ("weight_kg", "weight", "peso", "peso_kg"),
    "body_fat_pct": ("body_fat_pct", "grasa", "grasa_corporal", "body_fat"),
    "water_pct": ("water_pct", "agua", "water"),
    "muscle_mass_kg": ("muscle_mass_kg", "musculo", "músculo", "muscle_mass"),
    "waist_cm": ("waist_cm", "cintura"),
    "hip_cm": ("hip_cm", "cadera"),
    "chest_cm": ("chest_cm", "pecho"),
    "abdomen_cm": ("abdomen_cm", "abdomen"),
    "note": ("note", "nota"),
}


def _normalize_keys(raw: dict[str, Any], aliases: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    if not raw:
        return {}
    normalized: dict[str, Any] = {}
    lower_key_map = {str(key).lower(): value for key, value in raw.items()}
    for canonical, keys in aliases.items():
        for key in keys:
            if key in lower_key_map and lower_key_map[key] not in (None, ""):
                normalized[canonical] = lower_key_map[key]
                break
    return normalized


def normalize_physical_payload(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    normalized = _normalize_keys(raw, _PHYSICAL_ALIASES)
    if "weight_kg" in normalized:
        try:
            normalized["weight_kg"] = float(str(normalized["weight_kg"]).replace(",", "."))
        except (TypeError, ValueError):
            pass
    if "sleep_hours" in normalized:
        try:
            normalized["sleep_hours"] = float(str(normalized["sleep_hours"]).replace(",", "."))
        except (TypeError, ValueError):
            pass
    return normalized or None


def normalize_body_measurement_payload(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return _normalize_keys(raw, _BODY_FIELD_ALIASES)


def extract_weight_kg_from_text(text: str) -> float | None:
    for pattern in (
        r"(-?\d+(?:[.,]\d{1,2})?)\s*(?:kg|kilos?|kgs)\b",
        r"\bpeso\s+(-?\d+(?:[.,]\d{1,2})?)\b",
        r"\bpesad[oa]\s+(?:a\s+)?(-?\d+(?:[.,]\d{1,2})?)\b",
        r"\bme\s+pese\s+a?\s*(-?\d+(?:[.,]\d{1,2})?)\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            continue
    return None


async def register_weight_kg(*, date: str, weight_kg: float, note: str = "") -> dict[str, Any]:
    """Guarda peso en Medidas (BodyMeasurement) y diario físico (HealthPhysicalLog)."""
    from atlas_client import create_body_measurement, update_health

    body = await create_body_measurement(date, weight_kg=weight_kg, note=note)
    health = await update_health(date, physical={"weight_kg": weight_kg, "note": note})
    return {"body_measurement": body, "health": health}
