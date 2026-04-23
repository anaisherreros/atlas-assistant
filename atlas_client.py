from __future__ import annotations

import os
from typing import Any

import httpx


async def get_dashboard() -> Any:
    base_url = os.environ["ATLAS_VITAL_URL"].rstrip("/")
    assistant_api_key = os.environ["ASSISTANT_API_KEY"]
    url = f"{base_url}/api/assistant/dashboard/"

    headers = {"X-Assistant-Key": assistant_api_key}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
