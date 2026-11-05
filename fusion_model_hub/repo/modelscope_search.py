import logging

import httpx

logger = logging.getLogger(__name__)

# ModelScope retired the public GET /api/v1/models search (404 since ~2025).
# The live search endpoint is PUT /api/v1/dolphin/models with a JSON body.
_MODELSCOPE_API = "https://www.modelscope.cn/api/v1/dolphin/models"
_SEARCH_TIMEOUT = 15.0


async def search_modelscope(
    query: str = "",
    task: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict:
    body: dict = {
        "PageNumber": page,
        "PageSize": page_size,
    }
    if query:
        body["Name"] = query
    if task:
        body["Task"] = task
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT) as client:
            resp = await client.put(_MODELSCOPE_API, json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        logger.error("ModelScope API unavailable")
        return {"items": [], "total": 0, "source": "modelscope"}
    except httpx.HTTPStatusError as e:
        logger.warning("ModelScope search failed: %d %s", e.response.status_code, e.response.text)
        return {"items": [], "total": 0, "source": "modelscope"}
    except Exception:
        logger.exception("ModelScope search error")
        return {"items": [], "total": 0, "source": "modelscope"}

    payload = data.get("Data", {}) if isinstance(data, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    model_block = payload.get("Model", {}) or {}
    models = model_block.get("Models", []) or []
    total = model_block.get("TotalCount", 0) or 0
    items = []
    for m in models:
        tasks = m.get("Tasks") or []
        items.append(
            {
                "name": m.get("Name", ""),
                "id": m.get("Id", ""),
                "task": (tasks[0].get("Name", "") if tasks else ""),
                "framework": m.get("Framework", ""),
                "source": "modelscope",
                "downloads": m.get("Downloads", 0),
                "creator": m.get("Path", ""),
            }
        )
    logger.info("ModelScope search: query=%s total=%s", query, total)
    return {"items": items, "total": total, "source": "modelscope"}
