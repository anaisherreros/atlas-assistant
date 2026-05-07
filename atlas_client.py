from __future__ import annotations

import os
from typing import Any

import httpx


def _build_url(path: str) -> str:
    base_url = os.environ["ATLAS_VITAL_URL"].rstrip("/")
    return f"{base_url}{path}"


def _auth_headers() -> dict[str, str]:
    assistant_api_key = os.environ["ASSISTANT_API_KEY"]
    return {"X-Assistant-Key": assistant_api_key}


async def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            _build_url(path),
            headers=_auth_headers(),
            params=params or {},
        )
        response.raise_for_status()
        return response.json()


async def _post(path: str, payload: dict[str, Any]) -> Any:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(_build_url(path), headers=_auth_headers(), json=payload)
        response.raise_for_status()
        return response.json()


async def get_dashboard() -> Any:
    dashboard = await _get("/api/assistant/dashboard/")
    tasks_today = await get_tasks_today()
    today_data = await get_today()
    if isinstance(dashboard, dict):
        dashboard["tasks_today"] = tasks_today
        dashboard["today"] = today_data
        return dashboard
    return {"dashboard": dashboard, "tasks_today": tasks_today, "today": today_data}


async def get_today() -> Any:
    return await _get("/api/assistant/today/")


async def get_desire_structure(desire_id: int) -> Any:
    return await _get(f"/api/assistant/desires/{desire_id}/structure/")


async def get_all_desires_full() -> Any:
    return await _get("/api/assistant/desires/full/")


async def get_calendar(start_date: str, end_date: str) -> Any:
    return await _get(
        "/api/assistant/calendar/",
        params={"start_date": start_date, "end_date": end_date},
    )


async def get_areas_full() -> Any:
    return await _get("/api/assistant/areas/full/")


async def get_relationships_full() -> Any:
    return await _get("/api/assistant/relationships/full/")


async def get_reviews_summary() -> Any:
    return await _get("/api/assistant/reviews/summary/")


async def get_finance_full() -> Any:
    return await _get("/api/assistant/finance/full/")


async def get_finance() -> Any:
    """Mes en curso: transacciones, totales, categorías y resumen (Atlas Vital)."""
    return await _get("/api/assistant/finance/")


async def get_tasks_today() -> Any:
    return await _get("/api/assistant/tasks/today/")


async def get_tasks_pending() -> Any:
    return await _get("/api/assistant/tasks/pending/")


async def create_task(
    title: str,
    due_date: str,
    description: str = "",
    priority: str = "medium",
    start_time: str | None = None,
    end_time: str | None = None,
    goal_id: int | None = None,
) -> Any:
    payload: dict[str, Any] = {
        "title": title,
        "due_date": due_date,
        "description": description,
        "priority": priority,
    }
    if start_time:
        payload["start_time"] = start_time
    if end_time:
        payload["end_time"] = end_time
    if goal_id is not None:
        payload["goal_id"] = goal_id
    return await _post("/api/assistant/tasks/create/", payload)


async def complete_task(task_id: int) -> Any:
    payload = {"task_id": task_id}
    return await _post("/api/assistant/tasks/complete/", payload)


async def create_desire(
    title: str,
    description: str = "",
    area: str = "",
) -> Any:
    payload = {
        "title": title,
        "description": description,
        "area": area,
    }
    return await _post("/api/assistant/desires/create/", payload)


async def log_habit(
    habit_id: int,
    date: str,
    completed: bool,
    note: str = "",
) -> Any:
    payload = {
        "habit_id": habit_id,
        "date": date,
        "completed": completed,
        "note": note,
    }
    return await _post("/api/assistant/habits/log/", payload)


async def log_health(
    date: str,
    physical: dict[str, Any] | int | None = None,
    emotional: dict[str, Any] | int | None = None,
    mental: dict[str, Any] | int | None = None,
) -> Any:
    payload: dict[str, Any] = {"date": date, "physical": None, "emotional": None, "mental": None}

    def _prune(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if v is not None}
        return value

    if physical is not None:
        payload["physical"] = _prune(physical)
    if emotional is not None:
        payload["emotional"] = _prune(emotional)
    if mental is not None:
        payload["mental"] = _prune(mental)
    return await _post("/api/assistant/health/log/", payload)


async def create_transaction(
    description: str,
    amount: float,
    transaction_type: str,
    date: str,
) -> Any:
    payload = {
        "description": description,
        "amount": amount,
        "transaction_type": transaction_type,
        "date": date,
    }
    return await _post("/api/assistant/finance/transaction/", payload)


async def create_goal(
    phase_id: int,
    title: str,
    start_date: str,
    end_date: str,
) -> Any:
    payload = {
        "phase_id": phase_id,
        "title": title,
        "start_date": start_date,
        "end_date": end_date,
    }
    return await _post("/api/assistant/goals/create/", payload)


async def create_habit(
    title: str,
    start_date: str,
    frequency_type: str = "daily",
    goal_id: int | None = None,
    **kwargs: Any,
) -> Any:
    payload: dict[str, Any] = {
        "title": title,
        "start_date": start_date,
        "frequency_type": frequency_type,
        "goal_id": goal_id,
    }
    payload.update(kwargs)
    return await _post("/api/assistant/habits/create/", payload)


async def update_habit(habit_id: int, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {"habit_id": habit_id}
    payload.update(kwargs)
    return await _post("/api/assistant/habits/update/", payload)


async def delete_habit(habit_id: int) -> Any:
    payload = {"habit_id": habit_id}
    return await _post("/api/assistant/habits/delete/", payload)


async def update_task(task_id: int, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {"task_id": task_id}
    payload.update(kwargs)
    return await _post("/api/assistant/tasks/update/", payload)


async def delete_task(task_id: int) -> Any:
    payload = {"task_id": task_id}
    return await _post("/api/assistant/tasks/delete/", payload)


async def update_desire(desire_id: int, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {"desire_id": desire_id}
    payload.update(kwargs)
    return await _post("/api/assistant/desires/update/", payload)


async def delete_desire(desire_id: int) -> Any:
    payload = {"desire_id": desire_id}
    return await _post("/api/assistant/desires/delete/", payload)


async def create_phase(
    desire_id: int,
    title: str,
    start_date: str,
    end_date: str,
) -> Any:
    payload = {
        "desire_id": desire_id,
        "title": title,
        "start_date": start_date,
        "end_date": end_date,
    }
    return await _post("/api/assistant/phases/create/", payload)


async def update_phase(phase_id: int, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {"phase_id": phase_id}
    payload.update(kwargs)
    return await _post("/api/assistant/phases/update/", payload)


async def delete_phase(phase_id: int) -> Any:
    payload = {"phase_id": phase_id}
    return await _post("/api/assistant/phases/delete/", payload)


async def update_goal(goal_id: int, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {"goal_id": goal_id}
    payload.update(kwargs)
    return await _post("/api/assistant/goals/update/", payload)


async def delete_goal(goal_id: int) -> Any:
    payload = {"goal_id": goal_id}
    return await _post("/api/assistant/goals/delete/", payload)


async def create_daily_review(
    date: str,
    day_score: int | None = None,
    mood: str = "",
    note: str = "",
) -> Any:
    payload = {
        "date": date,
        "day_score": day_score,
        "mood": mood,
        "note": note,
    }
    return await _post("/api/assistant/reviews/daily/create/", payload)


async def create_weekly_review(
    week_start: str,
    week_end: str,
    **kwargs: Any,
) -> Any:
    payload: dict[str, Any] = {
        "week_start": week_start,
        "week_end": week_end,
    }
    payload.update(kwargs)
    return await _post("/api/assistant/reviews/weekly/create/", payload)


async def create_monthly_review(
    year: int,
    month: int,
    **kwargs: Any,
) -> Any:
    payload: dict[str, Any] = {"year": year, "month": month}
    payload.update(kwargs)
    return await _post("/api/assistant/reviews/monthly/create/", payload)


async def get_last_daily_review() -> Any:
    return await _get("/api/assistant/reviews/daily/last/")


async def get_last_weekly_review() -> Any:
    return await _get("/api/assistant/reviews/weekly/last/")


async def create_relationship(
    name: str,
    relationship_type: str,
    notes: str = "",
) -> Any:
    payload = {
        "name": name,
        "relationship_type": relationship_type,
        "notes": notes,
    }
    return await _post("/api/assistant/relationships/create/", payload)


async def log_relationship(
    person_id: int,
    date: str,
    interaction_summary: str,
    feeling: str,
    note: str = "",
) -> Any:
    payload = {
        "person_id": person_id,
        "date": date,
        "interaction_summary": interaction_summary,
        "feeling": feeling,
        "note": note,
    }
    return await _post("/api/assistant/relationships/log/", payload)


async def log_self_relationship(
    date: str,
    self_feeling: str,
    things_i_like: str = "",
    working_on: str = "",
) -> Any:
    payload = {
        "date": date,
        "self_feeling": self_feeling,
        "things_i_like": things_i_like,
        "working_on": working_on,
    }
    return await _post("/api/assistant/relationships/self/log/", payload)


async def update_health(
    date: str,
    physical: int | None = None,
    emotional: int | None = None,
    mental: int | None = None,
) -> Any:
    payload = {
        "date": date,
        "physical": physical,
        "emotional": emotional,
        "mental": mental,
    }
    return await _post("/api/assistant/health/update/", payload)


async def log_exercise(
    date: str,
    exercise_type: str,
    duration_minutes: int,
    note: str = "",
) -> Any:
    payload = {
        "date": date,
        "exercise_type": exercise_type,
        "duration_minutes": duration_minutes,
        "note": note,
    }
    return await _post("/api/assistant/health/exercise/log/", payload)


async def delete_transaction(transaction_id: int) -> Any:
    payload = {"transaction_id": transaction_id}
    return await _post("/api/assistant/finance/transaction/delete/", payload)


async def create_patrimony_snapshot(date: str, **kwargs: Any) -> Any:
    payload: dict[str, Any] = {"date": date}
    payload.update(kwargs)
    return await _post("/api/assistant/finance/patrimony/snapshot/create/", payload)
