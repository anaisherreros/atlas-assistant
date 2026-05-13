from __future__ import annotations

from tools import ATLAS_TOOLS

ALL_TOOL_NAMES = {tool["name"] for tool in ATLAS_TOOLS}

AGENT_TOOL_NAMES: dict[str, set[str]] = {
    "personal": set(ALL_TOOL_NAMES),
    "coach": {
        "get_dashboard",
        "get_all_desires_full",
        "get_desire_structure",
        "get_areas_full",
        "get_relationships_full",
        "get_reviews_summary",
        "create_desire",
        "update_desire",
        "create_goal",
        "update_goal",
        "delete_goal",
        "create_daily_review",
        "create_weekly_review",
        "create_monthly_review",
        "log_self_relationship",
    },
    "performance": {
        "get_today",
        "get_calendar",
        "get_all_desires_full",
        "get_desire_structure",
        "create_goal",
        "update_goal",
        "create_habit",
        "update_habit",
        "log_habit_completion",
        "log_health",
        "update_health",
        "log_exercise",
    },
    "financial": {
        "get_finance",
        "get_finance_full",
        "get_all_desires_full",
        "get_desire_structure",
        "create_goal",
        "update_goal",
        "create_habit",
        "update_habit",
        "log_habit_completion",
        "create_transaction",
        "delete_transaction",
        "create_patrimony_snapshot",
    },
}

AGENT_ALIASES: dict[str, str] = {
    "nutritionist": "performance",
    "trainer": "performance",
}


def get_tools_for_agent(agent_key: str) -> list[dict[str, object]]:
    normalized = AGENT_ALIASES.get(agent_key, agent_key)
    allowed_names = AGENT_TOOL_NAMES.get(normalized, AGENT_TOOL_NAMES["personal"])
    return [tool for tool in ATLAS_TOOLS if tool["name"] in allowed_names]
