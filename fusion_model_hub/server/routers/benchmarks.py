from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException

from ..deps import SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("")
async def list_benchmarks(
    chip: str | None = None,
    model_id: str | None = None,
    quant: str | None = None,
    settings: SettingsDep = None,
):
    mlx_url = settings.mlx_url.rstrip("/")
    params: dict = {}
    if chip:
        params["chip"] = chip
    if model_id:
        params["model_id"] = model_id
    if quant:
        params["quant"] = quant

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{mlx_url}/v1/benchmarks", params=params)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("MLX benchmarks returned %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError as e:
        logger.error("MLX not available for benchmarks: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Benchmarks query failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}")
async def get_benchmark(
    model_id: str,
    chip: str | None = None,
    quant: str | None = None,
    settings: SettingsDep = None,
):
    mlx_url = settings.mlx_url.rstrip("/")
    params: dict = {}
    if chip:
        params["chip"] = chip
    if quant:
        params["quant"] = quant

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{mlx_url}/v1/benchmarks/{model_id}", params=params,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Benchmark not found for model")
            logger.warning("MLX benchmark returned %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError as e:
        logger.error("MLX not available for benchmark: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Benchmark query failed")
        raise HTTPException(status_code=500, detail=str(e))
