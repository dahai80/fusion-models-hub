from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..deps import SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    model_path: str | None = Field(None, description="Local model path on disk")
    hf_repo: str | None = Field(None, description="HuggingFace repo (org/name)")


@router.post("")
async def analyze_model(request: AnalyzeRequest, settings: SettingsDep):
    if not request.model_path and not request.hf_repo:
        raise HTTPException(status_code=400, detail="At least one of model_path or hf_repo is required")

    mlx_url = settings.mlx_url.rstrip("/")
    payload: dict = {}
    if request.model_path:
        payload["model_path"] = request.model_path
    if request.hf_repo:
        payload["hf_repo"] = request.hf_repo

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{mlx_url}/v1/analyze", json=payload)
            if resp.status_code == 200:
                logger.info("Model analysis completed: %s", request.hf_repo or request.model_path)
                return resp.json()
            logger.warning("MLX analyze returned %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError as e:
        logger.error("MLX not available for analyze: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Model analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
