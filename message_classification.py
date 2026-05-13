from __future__ import annotations

import re


def classify_context(text: str) -> str:
    """Precarga de contexto Atlas en el system prompt: none / today / finance / full."""
    normalized = text.lower().strip()
    if not normalized:
        return "none"

    full_markers = (
        "reflexionar",
        "reflexión",
        "reflexiona",
        "analiza mis",
        "analizar mis",
        "análisis de mis",
        "analisis de mis",
        "analiza mi vida",
        "analiza mi situación",
        "analiza mi situacion",
        "coaching",
        "mis metas",
        "mis objetivos",
        "planificación",
        "planeación",
        "planeacion",
        "planifica mi",
        "balance de vida",
        "panorama general",
        "todas mis áreas",
        "mi vida en general",
        "filosofía",
        "sentido de mi vida",
        "patrones en mi",
        "desarrollo personal",
        "ayúdame a pensar",
        "ayudame a pensar",
        "piensa conmigo sobre",
        "analiza mis finanzas",
        "analizar mis finanzas",
        "revisa mi situación financiera",
        "revisa mi situacion financiera",
    )
    if any(marker in normalized for marker in full_markers):
        return "full"

    finance_markers = (
        "gasolina",
        "combustible",
        "diesel",
        "repostaje",
        "he gastado",
        "llevo gastado",
        "llevo gasto",
        "gasté en",
        "gaste en",
        "gastado en",
        "gasto en ",
        "cuánto he gastado",
        "cuanto he gastado",
        "cuánto llevo gastado",
        "cuanto llevo gastado",
        "cuánto gasté",
        "cuanto gaste",
        "gastos del mes",
        "ingresos del mes",
        "balance del mes",
        "mis finanzas",
        "situación financiera",
        "situacion financiera",
        "transacciones del mes",
        "movimientos del mes",
        "cuánto dinero",
        "cuanto dinero",
        "desglose por categoría",
        "desglose por categoria",
        "presupuesto del mes",
    )
    if any(marker in normalized for marker in finance_markers):
        return "finance"
    if re.search(
        r"\b(cuánto|cuanto)\b.*\b(gastado|gastos|gasté|gaste)\b",
        normalized,
    ):
        return "finance"
    if re.search(r"\b(gasolina|combustible|diesel)\b", normalized) and re.search(
        r"\b(cuánto|cuanto|qué|que|llevo|coste|costo)\b",
        normalized,
    ):
        return "finance"

    if re.fullmatch(
        r"(hola|hey|buenas|buenos días|buenas tardes|buenas noches)(\s*[!.¡…]*)?",
        normalized,
    ):
        return "none"
    if re.fullmatch(
        r"(hola|hey)\s*,?\s*(qué|que)\s+tal\s*[!.¡?¿]*",
        normalized,
    ):
        return "none"
    thanks_only = ("gracias", "muchas gracias", "ok", "vale", "perfecto", "genial")
    if normalized in thanks_only:
        return "none"

    conceptual_starts = (
        "qué es ",
        "que es ",
        "qué son ",
        "que son ",
        "define ",
        "define qué ",
        "define que ",
    )
    if any(normalized.startswith(prefix) for prefix in conceptual_starts):
        if " mi " not in normalized and not normalized.startswith("mi "):
            return "none"

    today_markers = (
        "qué tengo hoy",
        "que tengo hoy",
        "qué tengo para hoy",
        "que tengo para hoy",
        "marcar",
        "completar",
        "check",
        "registra",
        "apunta",
        "cuánto llevo",
        "cuanto llevo",
        "mis hábitos",
        "mis habitos",
        "mis tareas",
        "crea una tarea",
        "crea tarea",
        "nueva tarea",
        "tarea para",
        "marca el hábito",
        "marca el habito",
        "hábito",
        "habito",
        "hábitos",
        "habitos",
        "para hoy",
        "mi día",
        "mi dia",
        "calendario",
        "agenda",
        "entre fechas",
        "rango de fechas",
        "estructura del deseo",
        "estructura de mi deseo",
        "estructura de un deseo",
        "todos los deseos",
        "mis deseos activos",
        "deseos activos",
        "listado de deseos",
        "mis áreas",
        "areas de vida",
        "áreas de vida",
        "subáreas",
        "subareas",
        "mis relaciones",
        "relaciones personales",
        "historial de relaciones",
        "resumen de revisiones",
        "revisiones diaria",
        "revision semanal",
        "revisión mensual",
        "revision mensual",
        "finanzas completas",
        "presupuesto anual",
        "gastos del mes",
        "gastos reales",
        "crea un deseo",
        "crea deseo",
    )
    if any(marker in normalized for marker in today_markers):
        return "today"

    if len(normalized.split()) <= 14 and re.match(
        r"^(crea|haz|marca|completa|registra|apunta|muestra|dime)\s+",
        normalized,
    ):
        return "today"

    if re.search(r"\b(mi|mis|me)\s+", normalized):
        return "full"

    if len(normalized.split()) > 25:
        return "full"

    return "none"


def classify_message(text: str) -> str:
    normalized = text.lower().strip()
    simple_keywords = (
        "qué tengo hoy",
        "que tengo hoy",
        "marcar",
        "completar",
        "check",
        "registra",
        "apunta",
        "cuánto llevo",
        "cuanto llevo",
        "mis hábitos",
        "mis habitos",
        "mis tareas",
        "estructura del deseo",
        "estructura de mi deseo",
        "estructura de un deseo",
        "estructura completa del deseo",
        "objetivos del deseo",
        "todos los deseos",
        "mis deseos activos",
        "deseos activos",
        "deseos completos",
        "listado de deseos",
        "calendario",
        "en el calendario",
        "entre fechas",
        "rango de fechas",
        "agenda entre",
        "mis áreas",
        "areas de vida",
        "áreas de vida",
        "subáreas",
        "subareas",
        "mis relaciones",
        "relaciones personales",
        "historial de relaciones",
        "resumen de revisiones",
        "revisiones diaria",
        "revision semanal",
        "revisión mensual",
        "revision mensual",
        "finanzas completas",
        "presupuesto anual",
        "gastos del mes",
        "gastos reales",
        "gasolina",
        "combustible",
        "he gastado",
        "llevo gastado",
        "cuánto he gastado",
        "cuanto he gastado",
        "mis finanzas",
        "balance del mes",
        "ingresos del mes",
    )
    if any(keyword in normalized for keyword in simple_keywords):
        return "simple"

    words = [word for word in normalized.split() if word]
    if len(words) < 15:
        direct_starts = (
            "que ",
            "qué ",
            "cuanto ",
            "cuánto ",
            "marca ",
            "completa ",
            "registra ",
            "apunta ",
            "crea ",
            "haz ",
            "muestra ",
            "dime ",
        )
        if normalized.endswith("?") or normalized.startswith(direct_starts):
            return "simple"

    return "complex"
