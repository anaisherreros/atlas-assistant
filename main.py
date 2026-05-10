from __future__ import annotations

import json
import logging
import os
import re

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agents import build_agent_system_prompt, get_agent
from atlas_client import get_dashboard, get_finance, get_today
from claude import generate_with_tools
from database import (
    create_engine,
    fetch_conversation_messages,
    get_active_agent,
    init_db,
    messages_to_anthropic,
    save_message,
    session_factory,
    set_active_agent,
)
from router import TRANSITION_MESSAGES, detect_agent

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"
MAX_HISTORY_MESSAGES = 20
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_TOOL_LOOPS = 12


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def classify_context(text: str) -> str:
    """Precarga de contexto Atlas en el system prompt: none / today / full."""
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
    if any(m in normalized for m in full_markers):
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
    if any(m in normalized for m in finance_markers):
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
    if any(normalized.startswith(s) for s in conceptual_starts):
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
    if any(m in normalized for m in today_markers):
        return "today"

    if len(normalized.split()) <= 14:
        if re.match(
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


async def post_init(application: Application) -> None:
    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    await init_db(engine)
    application.bot_data["engine"] = engine
    application.bot_data["session_factory"] = session_factory(engine)
    application.bot_data["anthropic"] = AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    logger.info("Base de datos lista y cliente Anthropic configurado.")


async def post_shutdown(application: Application) -> None:
    engine = application.bot_data.get("engine")
    if engine is not None:
        await engine.dispose()
    logger.info("Motor SQLAlchemy cerrado.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_chat is None:
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id if update.effective_user else chat_id

    session_factory_: async_sessionmaker[AsyncSession] = context.application.bot_data[
        "session_factory"
    ]
    client: AsyncAnthropic = context.application.bot_data["anthropic"]

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    async with session_factory_() as session:
        previous_agent = await get_active_agent(session, telegram_chat_id=chat_id)
        selected_agent = detect_agent(text, previous_agent)
        if selected_agent != previous_agent:
            transition = TRANSITION_MESSAGES.get(
                selected_agent,
                "Cambiando de agente...",
            )
            await update.message.reply_text(transition)
            await set_active_agent(
                session,
                telegram_chat_id=chat_id,
                active_agent=selected_agent,
            )
            logger.info(
                "Agente activo: %s → %s",
                previous_agent,
                selected_agent,
            )

        agent = get_agent(selected_agent)

        ctx = classify_context(text)
        if ctx == "none":
            history_limit = 0
        elif ctx == "today":
            history_limit = 5
        elif ctx == "finance":
            history_limit = 8
        else:
            history_limit = MAX_HISTORY_MESSAGES

        history_rows = await fetch_conversation_messages(
            session,
            telegram_chat_id=chat_id,
            limit=history_limit,
        )
        api_messages = messages_to_anthropic(history_rows)
        if ctx == "none":
            api_messages = [{"role": "user", "content": text}]
        else:
            api_messages.append({"role": "user", "content": text})

        dashboard_data = "{}"
        logger.info(
            "Contexto Atlas (precarga): %s | agente: %s",
            ctx,
            selected_agent,
        )
        logger.info("Historial chat: limit=%s (mensajes=%d)", history_limit, len(api_messages))

        try:
            if ctx == "full":
                dashboard = await get_dashboard()
                dashboard_data = json.dumps(
                    dashboard, ensure_ascii=False, separators=(",", ":")
                )
            elif ctx == "today":
                dashboard = await get_today()
                dashboard_data = json.dumps(
                    dashboard, ensure_ascii=False, separators=(",", ":")
                )
            elif ctx == "finance":
                dashboard = await get_finance()
                dashboard_data = json.dumps(
                    dashboard, ensure_ascii=False, separators=(",", ":")
                )
        except Exception:
            logger.exception("Error al consultar Atlas Vital")

        system_prompt = build_agent_system_prompt(agent, dashboard_data)

        try:
            complexity = classify_message(text)
            model = MODEL
            logger.info("Clasificacion de mensaje: %s (modelo: %s)", complexity, model)
            logger.info("Modelo elegido: %s para: %s", model, text[:50])
            assistant_text, tools_used = await generate_with_tools(
                client,
                model=model,
                system_prompt=system_prompt,
                api_messages=api_messages,
                max_tool_loops=MAX_TOOL_LOOPS,
            )
        except Exception:
            logger.exception("Error al llamar a la API de Anthropic")
            await update.message.reply_text(
                "No pude obtener respuesta del asistente ahora mismo. "
                "Inténtalo de nuevo en unos segundos."
            )
            return

        response_words = len(assistant_text.split())
        is_action = tools_used and response_words < 100
        if not is_action:
            await save_message(
                session,
                telegram_chat_id=chat_id,
                telegram_user_id=user_id,
                role="user",
                content=text,
            )
            await save_message(
                session,
                telegram_chat_id=chat_id,
                telegram_user_id=user_id,
                role="assistant",
                content=assistant_text,
            )

    for part in chunk_text(assistant_text, TELEGRAM_MAX_MESSAGE_LENGTH):
        await update.message.reply_text(part)


def main() -> None:
    required = (
        "TELEGRAM_BOT_TOKEN",
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "ATLAS_VITAL_URL",
        "ASSISTANT_API_KEY",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Faltan variables de entorno obligatorias: " + ", ".join(missing)
        )

    token = os.environ["TELEGRAM_BOT_TOKEN"]

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
