import logging

import httpx

logger = logging.getLogger(__name__)

_MODELSCOPE_API = "https://modelscope.cn/api/v1/models"
_SEARCH_TIMEOUT = 15.0


async def search_modelscope(
    query: str = "",
    task: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict:
    params: dict = {
        "PageNumber": page,
        "PageSize": page_size,
    }
    if query:
        params["Name"] = query
    if task:
        params["Task"] = task
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT) as client:
            resp = await client.get(_MODELSCOPE_API, params=params)
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

    models = data.get("Data", {}).get("Models", []) if isinstance(data, dict) else []
    total = data.get("Data", {}).get("TotalCount", 0) if isinstance(data, dict) else 0
    items = []
    for m in models:
        items.append(
            {
                "name": m.get("Name", ""),
                "id": m.get("Id", ""),
                "task": m.get("Task", ""),
                "framework": m.get("Framework", ""),
                "source": "modelscope",
                "downloads": m.get("Downloads", 0),
                "creator": m.get("Creator", ""),
            }
        )
    return {"items": items, "total": total, "source": "modelscope"}
