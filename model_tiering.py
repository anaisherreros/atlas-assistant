from __future__ import annotations

import os
import re
import unicodedata

DEFAULT_MODEL_FAST = "claude-haiku-4-5"
DEFAULT_MODEL_SMART = "claude-sonnet-4-5"
DEFAULT_REFLECTIVE_WORD_THRESHOLD = 80

DEFAULT_REFLECTIVE_KEYWORDS: tuple[str, ...] = (
    "por qué",
    "porque",
    "siento",
    "me siento",
    "revisión",
    "revision",
    "analiza",
    "analizar",
    "ayúdame a pensar",
    "ayudame a pensar",
    "reflexiona",
    "reflexionar",
    "patrón",
    "patron",
    "bloqueo",
    "bloqueos",
    "por qué hago",
    "qué significa",
    "que significa",
    "no entiendo por qué",
    "hablar de",
    "necesito pensar",
)


def get_model_fast() -> str:
    return os.getenv("MODEL_FAST", DEFAULT_MODEL_FAST).strip() or DEFAULT_MODEL_FAST


def get_model_smart() -> str:
    return os.getenv("MODEL_SMART", DEFAULT_MODEL_SMART).strip() or DEFAULT_MODEL_SMART


def get_reflective_word_threshold() -> int:
    raw = os.getenv("REFLECTIVE_WORD_THRESHOLD", str(DEFAULT_REFLECTIVE_WORD_THRESHOLD))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_REFLECTIVE_WORD_THRESHOLD


def get_reflective_keywords() -> tuple[str, ...]:
    raw = os.getenv("REFLECTIVE_KEYWORDS", "").strip()
    if not raw:
        return DEFAULT_REFLECTIVE_KEYWORDS
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def select_model_for_message(*, agent_key: str, user_text: str) -> tuple[str, str]:
    """Elige MODEL_FAST o MODEL_SMART con reglas locales (sin llamada LLM extra)."""
    smart = get_model_smart()
    fast = get_model_fast()

    if agent_key == "coach":
        return smart, "agent_coach"

    word_count = count_words(user_text)
    threshold = get_reflective_word_threshold()
    if word_count > threshold:
        return smart, f"long_message_{word_count}_words"

    normalized = _normalize_for_match(user_text)
    for keyword in get_reflective_keywords():
        if _normalize_for_match(keyword) in normalized:
            return smart, f"reflective_keyword:{keyword}"

    return fast, "default_fast"
