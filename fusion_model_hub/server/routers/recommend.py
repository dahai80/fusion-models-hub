from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...recommend.engine import RecommendEngine
from ...recommend.types import RecommendRequest, RecommendResponse
from ..deps import SessionDep, SettingsDep

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
        raise HTTPException(status_code=503, detail=f"Recommendation unavailable: {e}")


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
        raise HTTPException(status_code=503, detail=f"Recommendation unavailable: {e}")


async def _fetch_models_from_db(session, request: RecommendRequest) -> list[dict]:
    try:
        from ...db.crud import list_models
        result, _ = await list_models(session, page_size=200)
        return [
            {
                "id": str(m.id),
                "model_id": m.name,
                "name": m.name,
                "params_b": 0,
                "task": "llm",
                "quant_type": "Q4_K_M",
                "download_count": 0,
            }
            for m in result
        ]
    except Exception as e:
        logger.warning("Failed to fetch models from DB: %s", e)
        return []
