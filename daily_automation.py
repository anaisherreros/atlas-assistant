from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from telegram.ext import Application

from atlas_client import get_last_daily_review, get_today
from database import fetch_known_chat_ids
from deterministic_handlers import _extract_habits_from_payload, _extract_tasks_from_payload

logger = logging.getLogger(__name__)

AUTOMATION_TASKS_KEY = "daily_automation_tasks"
DEFAULT_TIMEZONE = "Europe/Zurich"
DEFAULT_MORNING_TIME = "07:30"
DEFAULT_NIGHT_TIME = "21:30"
TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _normalize_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _get_timezone() -> ZoneInfo:
    timezone_name = os.getenv("AUTOMATION_TIMEZONE", DEFAULT_TIMEZONE)
    return ZoneInfo(timezone_name)


def _parse_clock(value: str | None, default_value: str) -> time:
    raw = (value or default_value).strip()
    hour_str, minute_str = raw.split(":", maxsplit=1)
    return time(hour=int(hour_str), minute=int(minute_str))


def _next_run(now: datetime, target: time) -> datetime:
    scheduled = now.replace(
        hour=target.hour,
        minute=target.minute,
        second=0,
        microsecond=0,
    )
    if scheduled <= now:
        scheduled += timedelta(days=1)
    return scheduled


def _sort_tasks(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(task: dict[str, object]) -> tuple[int, str, str]:
        start_time = str(task.get("start_time") or "")
        has_time = 0 if start_time else 1
        return (has_time, start_time, str(task.get("title") or ""))

    return sorted(tasks, key=sort_key)


def _format_task_line(task: dict[str, object]) -> str:
    task_id = task.get("id")
    title = str(task.get("title") or "Sin titulo")
    start_time = str(task.get("start_time") or "").strip()
    end_time = str(task.get("end_time") or "").strip()
    time_block = ""
    if start_time and end_time:
        time_block = f"{start_time}-{end_time}"
    elif start_time:
        time_block = start_time
    id_block = f"[{task_id}] " if task_id is not None else ""
    if time_block:
        return f"- {time_block} · {id_block}{title}"
    return f"- {id_block}{title}"


def _format_habit_line(habit: dict[str, object]) -> str:
    habit_id = habit.get("id")
    title = str(habit.get("title") or "Sin titulo")
    completed = habit.get("completed")
    status = str(habit.get("status") or "")
    is_done = completed in (True, 1, "true", "True") or status == "completed"
    marker = "x" if is_done else " "
    id_block = f"[{habit_id}] " if habit_id is not None else ""
    return f"- [{marker}] {id_block}{title}"


def _review_url() -> str:
    return os.getenv("DAILY_REVIEW_URL") or os.environ["ATLAS_VITAL_URL"].rstrip("/")


def _extract_review_date(value: object) -> str | None:
    if isinstance(value, dict):
        for key in ("date", "review_date", "created_at", "updated_at"):
            raw = value.get(key)
            if isinstance(raw, str) and len(raw) >= 10:
                return raw[:10]
        for child in value.values():
            review_date = _extract_review_date(child)
            if review_date:
                return review_date
    elif isinstance(value, list):
        for child in value:
            review_date = _extract_review_date(child)
            if review_date:
                return review_date
    return None


def _is_review_done_today(review_payload: object, today_iso: str) -> bool:
    review_date = _extract_review_date(review_payload)
    return review_date == today_iso


def _build_morning_message(today_payload: object) -> str:
    tasks = _sort_tasks(_extract_tasks_from_payload(today_payload))
    habits = _extract_habits_from_payload(today_payload)

    timed_tasks = [task for task in tasks if task.get("start_time")]
    untimed_tasks = [task for task in tasks if not task.get("start_time")]
    pending_habits = [
        habit
        for habit in habits
        if habit.get("completed") not in (True, 1, "true", "True")
        and str(habit.get("status") or "") != "completed"
    ]

    lines = ["Buenos dias. Esto tienes hoy:"]
    lines.append("")
    lines.append(f"Tareas de hoy: {len(tasks)}")
    if timed_tasks:
        lines.append("Con hora:")
        lines.extend(_format_task_line(task) for task in timed_tasks[:10])
    if untimed_tasks:
        lines.append("Sin hora:")
        lines.extend(_format_task_line(task) for task in untimed_tasks[:10])
    if not tasks:
        lines.append("- No veo tareas para hoy.")

    lines.append("")
    lines.append(f"Habitos de hoy: {len(habits)}")
    if habits:
        lines.extend(_format_habit_line(habit) for habit in habits[:12])
    else:
        lines.append("- No veo habitos para hoy.")

    if pending_habits:
        lines.append("")
        lines.append(f"Pendientes ahora: {len(pending_habits)} habitos.")

    return "\n".join(lines)


def _build_night_message(today_payload: object, *, review_done: bool, review_url: str) -> str:
    tasks = _extract_tasks_from_payload(today_payload)
    habits = _extract_habits_from_payload(today_payload)

    completed_tasks = [
        task
        for task in tasks
        if str(task.get("status") or "") in {"completed", "done"}
        or task.get("completed") in (True, 1, "true", "True")
    ]
    pending_tasks = [task for task in tasks if task not in completed_tasks]

    completed_habits = [
        habit
        for habit in habits
        if habit.get("completed") in (True, 1, "true", "True")
        or str(habit.get("status") or "") == "completed"
    ]
    pending_habits = [habit for habit in habits if habit not in completed_habits]

    lines = ["Cierre del dia:"]
    lines.append(f"- Tareas completadas: {len(completed_tasks)}/{len(tasks)}")
    lines.append(f"- Habitos marcados: {len(completed_habits)}/{len(habits)}")

    if pending_tasks:
        lines.append("")
        lines.append("Te quedan tareas abiertas:")
        lines.extend(_format_task_line(task) for task in _sort_tasks(pending_tasks)[:8])

    if pending_habits:
        lines.append("")
        lines.append("Habitos sin marcar:")
        lines.extend(_format_habit_line(habit) for habit in pending_habits[:8])

    lines.append("")
    if review_done:
        lines.append("La review diaria de hoy ya esta hecha.")
    else:
        lines.append(f"Te falta la review diaria. Hazla aqui: {review_url}")

    return "\n".join(lines)


async def _send_text(application: Application, chat_id: int, text: str) -> None:
    for part in chunk_text(text, TELEGRAM_MAX_MESSAGE_LENGTH):
        await application.bot.send_message(chat_id=chat_id, text=part)


async def _broadcast_daily_message(
    application: Application,
    *,
    message_builder,
) -> None:
    session_factory_: async_sessionmaker[AsyncSession] = application.bot_data["session_factory"]

    async with session_factory_() as session:
        chat_ids = await fetch_known_chat_ids(session)
    if not chat_ids:
        logger.info("Automatizacion diaria: no hay chats registrados.")
        return

    today_payload = await get_today()
    message = await message_builder(today_payload)
    for chat_id in chat_ids:
        try:
            await _send_text(application, chat_id, message)
        except Exception:
            logger.exception("No pude enviar automatizacion al chat %s", chat_id)


async def _build_night_message_async(today_payload: object) -> str:
    timezone = _get_timezone()
    today_iso = datetime.now(timezone).date().isoformat()
    review_payload = await get_last_daily_review()
    return _build_night_message(
        today_payload,
        review_done=_is_review_done_today(review_payload, today_iso),
        review_url=_review_url(),
    )


async def _run_morning_brief(application: Application) -> None:
    logger.info("Ejecutando automatizacion matinal.")
    await _broadcast_daily_message(
        application,
        message_builder=lambda payload: asyncio.sleep(0, result=_build_morning_message(payload)),
    )


async def _run_night_recap(application: Application) -> None:
    logger.info("Ejecutando automatizacion nocturna.")
    await _broadcast_daily_message(
        application,
        message_builder=_build_night_message_async,
    )


async def _daily_loop(
    application: Application,
    *,
    task_name: str,
    schedule_time: time,
    callback,
) -> None:
    timezone = _get_timezone()
    while True:
        now = datetime.now(timezone)
        next_run = _next_run(now, schedule_time)
        wait_seconds = max((next_run - now).total_seconds(), 1.0)
        logger.info(
            "Automatizacion %s programada para %s",
            task_name,
            next_run.isoformat(),
        )
        await asyncio.sleep(wait_seconds)
        try:
            await callback(application)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Fallo en automatizacion %s", task_name)


def start_daily_automation(application: Application) -> None:
    if not _normalize_bool(os.getenv("ENABLE_DAILY_AUTOMATIONS"), default=True):
        logger.info("Automatizaciones diarias desactivadas por entorno.")
        return

    morning_time = _parse_clock(os.getenv("MORNING_BRIEF_TIME"), DEFAULT_MORNING_TIME)
    night_time = _parse_clock(os.getenv("NIGHT_RECAP_TIME"), DEFAULT_NIGHT_TIME)

    tasks = [
        asyncio.create_task(
            _daily_loop(
                application,
                task_name="morning_brief",
                schedule_time=morning_time,
                callback=_run_morning_brief,
            )
        ),
        asyncio.create_task(
            _daily_loop(
                application,
                task_name="night_recap",
                schedule_time=night_time,
                callback=_run_night_recap,
            )
        ),
    ]
    application.bot_data[AUTOMATION_TASKS_KEY] = tasks
    logger.info("Automatizaciones diarias iniciadas.")


async def stop_daily_automation(application: Application) -> None:
    tasks = application.bot_data.pop(AUTOMATION_TASKS_KEY, [])
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
