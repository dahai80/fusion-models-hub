import logging
import time
from datetime import UTC, datetime

from fastapi import APIRouter

from ...db import crud
from ..deps import get_session_factory
from .inference import _loaded_models, _model_stats

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitor"])


@router.get("/monitor/realtime")
async def realtime_monitor():
    sf = get_session_factory()
    models_data = []
    loaded_count = 0
    total_concurrent = 0
    total_requests_today = 0
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    async with sf() as session:
        all_models, _ = await crud.list_models(session, page_size=10000)
        model_map = {m.id: m for m in all_models}

        for model_id, model_obj in model_map.items():
            info = _loaded_models.get(model_id)
            stats = _model_stats.get(model_id, {})
            is_loaded = info is not None
            if is_loaded:
                loaded_count += 1

            entry = {
                "model_id": model_id,
                "model_name": model_obj.name,
                "status": "loaded" if is_loaded else "not_loaded",
                "pinned": model_obj.pinned,
                "concurrent_requests": 0,
                "tokens_per_second": 0.0,
                "source_module": "",
                "avg_latency_ms": 0.0,
                "total_requests": stats.get("request_count", 0),
                "total_tokens": stats.get("total_tokens", 0),
                "memory_usage_mb": 0,
                "loaded_since": None,
                "last_request_at": None,
                "running_node": "local",
            }

            if stats.get("request_count", 0) > 0:
                entry["avg_latency_ms"] = round(
                    stats.get("total_latency", 0.0) / stats["request_count"], 2
                )
                last_at = stats.get("last_request_at", 0)
                if last_at and last_at > 0:
                    elapsed = time.time() - last_at
                    if elapsed < 60:
                        total_tokens = stats.get("total_tokens", 0)
                        first_at = stats.get("first_request_at", last_at)
                        duration = max(last_at - first_at, 1.0)
                        entry["tokens_per_second"] = round(total_tokens / duration, 2)
                        entry["concurrent_requests"] = 1
                source_mod = stats.get("source_module", "")
                if source_mod:
                    entry["source_module"] = source_mod

            if stats.get("last_request_at"):
                entry["last_request_at"] = datetime.fromtimestamp(
                    stats["last_request_at"], tz=UTC
                ).isoformat()
                if stats["last_request_at"] >= today_start:
                    total_requests_today += stats.get("request_count", 0)

            if is_loaded:
                entry["loaded_since"] = datetime.fromtimestamp(
                    info.get("loaded_at", 0), tz=UTC
                ).isoformat()
                v = await crud.get_version(session, info.get("version_id", ""))
                if v and v.memory_usage > 0:
                    entry["memory_usage_mb"] = v.memory_usage

            models_data.append(entry)

    summary = {
        "loaded_count": loaded_count,
        "total_concurrent": total_concurrent,
        "total_requests_today": total_requests_today,
    }
    return {"models": models_data, "summary": summary}
