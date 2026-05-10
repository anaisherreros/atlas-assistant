from __future__ import annotations

VALID_AGENT_KEYS = frozenset({"personal", "coach", "nutritionist", "trainer"})

TRANSITION_MESSAGES: dict[str, str] = {
    "coach": "Conectando con tu Coach... 🧘",
    "nutritionist": "Conectando con tu nutricionista... 🥗",
    "trainer": "Conectando con tu entrenador... 💪",
    "personal": "Volviendo con tu asistente personal... ✨",
}


def detect_agent(message: str, current_agent: str) -> str:
    msg = message.lower()
    if current_agent not in VALID_AGENT_KEYS:
        current_agent = "personal"

    if "pasame con" in msg or "pásame con" in msg or "habla con" in msg:
        if "coach" in msg:
            return "coach"
        if "nutricion" in msg:
            return "nutritionist"
        if "entrenador" in msg or "entrenadora" in msg:
            return "trainer"
        if "asistente" in msg or "vuelve" in msg:
            return "personal"

    nutrition_keywords = (
        "comer",
        "comida",
        "menú",
        "menu",
        "macros",
        "calorías",
        "calorias",
        "proteína",
        "proteina",
        "nutrición",
        "nutricion",
        "dieta",
    )
    trainer_keywords = (
        "entreno",
        "ejercicio",
        "gym",
        "gimnasio",
        "rutina",
        "series",
        "peso",
        "pesas",
        "cardio",
    )
    coach_keywords = (
        "meta",
        "objetivo",
        "bloqueo",
        "propósito",
        "proposito",
        "motivación",
        "motivacion",
        "reflexión",
        "reflexion",
        "valores",
        "ikigai",
    )

    if any(k in msg for k in nutrition_keywords):
        return "nutritionist"
    if any(k in msg for k in trainer_keywords):
        return "trainer"
    if any(k in msg for k in coach_keywords):
        return "coach"

    return current_agent
