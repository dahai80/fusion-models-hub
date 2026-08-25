from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...recommend.engine import RecommendEngine
from ...recommend.types import RecommendRequest, RecommendResponse
from ..deps import SessionDep, SettingsDep
from ..errors import safe_http_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["recommend"])

_engine: RecommendEngine | None = None


def _get_engine(settings: SettingsDep) -> RecommendEngine:
    global _engine
    # E-E10: invalidate on mlx_url OR mlx_internal_api_key drift, not just mlx_url.
    # On rebuild, drop the old engine's embedded HardwareDetector 5-min cache so a
    # hot-reload-swapped MLX URL does not keep serving stale hardware.
    if (
        _engine is None
        or _engine.mlx_url != settings.mlx_url
        or _engine.api_key != settings.mlx_internal_api_key
    ):
        if _engine is not None:
            _engine.invalidate_cache()
        _engine = RecommendEngine(settings.mlx_url, api_key=settings.mlx_internal_api_key)
        logger.info(
            "RecommendEngine (re)built for mlx_url=%s",
            settings.mlx_url,
        )
    return _engine


class QuickRecommendRequest(BaseModel):
    task: str = Field("llm", description="Task type: llm|text2image|text2video|embedding")
    preference: str = Field("balanced", description="quality|balanced|speed")
    max_results: int = Field(10, ge=1, le=50)


@router.post("")
async def recommend_models(request: RecommendRequest, settings: SettingsDep, session: SessionDep) -> RecommendResponse:
    engine = _get_engine(settings)
    models_from_db = await _fetch_models_from_db(session, request)
    try:
        return await engine.recommend(request, models_from_db)
    except Exception as e:
        logger.error("Recommendation engine failed: %s", e)
        raise safe_http_error(503, "Recommendation unavailable", exc=e, context="recommend")


@router.get("/quick")
async def quick_recommend(
    task: str = "llm",
    preference: str = "balanced",
    max_results: int = 10,
    settings: SettingsDep = None,
    session: SessionDep = None,
):
    engine = _get_engine(settings)
    request = RecommendRequest(task=task, preference=preference, max_results=max_results)
    models_from_db = await _fetch_models_from_db(session, request)
    try:
        return await engine.recommend(request, models_from_db)
    except Exception as e:
        logger.error("Quick recommend failed: %s", e)
        raise safe_http_error(503, "Recommendation unavailable", exc=e, context="quick-recommend")


def _parse_params_b(size: str | None) -> float:
    # E-E3: feed the RecommendEngine real params_size instead of a hardcoded 0.
    # "7B" -> 7.0, "700M" -> 0.7, "3500M" -> 3.5; unparseable -> 0.0 so it falls
    # through the min/max params filter as a no-op rather than being dropped.
    if not size:
        return 0.0
    try:
        s = size.lower().strip()
        if s.endswith("b"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) / 1000
        return float(s)
    except (ValueError, TypeError):
        return 0.0


async def _fetch_models_from_db(session, request: RecommendRequest) -> list[dict]:
    try:
        from ...db.crud import list_models
        result, _ = await list_models(session, page_size=200)
        # E-E3: previously hardcoded params_b:0, task:"llm", download_count:0
        # for every model — so the RecommendEngine's params filter (engine.py
        # `min_params_b <= params_b <= max_params_b`) rejected all candidates
        # whenever min_params_b>0, and download_count never influenced ranking.
        # Read the real columns: params_size -> params_b, task_types -> task,
        # download_count. task_types is comma-separated; take the first entry,
        # defaulting to "llm".
        return [
            {
                "id": str(m.id),
                "model_id": m.name,
                "name": m.name,
                "params_b": _parse_params_b(m.params_size),
                "task": ((m.task_types or "").split(",")[0].strip() or "llm"),
                "quant_type": "Q4_K_M",
                "download_count": m.download_count or 0,
            }
            for m in result
        ]
    except Exception as e:
        logger.warning("Failed to fetch models from DB: %s", e)
        return []
