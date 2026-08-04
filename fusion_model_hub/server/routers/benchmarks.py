from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..deps import SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


class BenchTriggerRequest(BaseModel):
    model_id: str = ""
    suite: str = "general"
    callback_url: str = ""


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


@router.get("/compare")
async def compare_benchmarks(
    model_ids: str = "",
    chip: str | None = None,
    settings: SettingsDep = None,
):
    if not model_ids:
        raise HTTPException(status_code=400, detail="model_ids is required (comma-separated)")
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="model_ids is required (comma-separated)")
    mlx_url = settings.mlx_url.rstrip("/")
    params: dict = {}
    if chip:
        params["chip"] = chip
    results: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for mid in ids:
                resp = await client.get(f"{mlx_url}/v1/benchmarks/{mid}", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for item in data:
                            item.setdefault("model_id", mid)
                        results.extend(data)
                    else:
                        entry = {**data, "model_id": mid} if isinstance(data, dict) else {"model_id": mid, "data": data}
                        results.append(entry)
                else:
                    logger.warning("MLX benchmark compare: %s returned %d", mid, resp.status_code)
                    results.append({"model_id": mid, "error": f"status {resp.status_code}"})
    except httpx.ConnectError as e:
        logger.error("MLX not available for benchmark compare: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except Exception as e:
        logger.exception("Benchmark compare failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"items": results, "model_ids": ids}


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


@router.post("/trigger")
async def trigger_benchmark(body: BenchTriggerRequest, settings: SettingsDep):
    bench_url = settings.bench_url.rstrip("/")
    payload = {
        "suite": body.suite,
        "model_id": body.model_id,
    }
    if body.callback_url:
        payload["callback_url"] = body.callback_url
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{bench_url}/api/v1/tasks", json=payload)
            if resp.status_code in (200, 201, 202):
                logger.info("Bench trigger submitted: model=%s suite=%s", body.model_id, body.suite)
                return {"status": "submitted", "detail": resp.json()}
            logger.warning("Bench trigger returned %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError:
        logger.error("Fusion-Bench not available at %s", bench_url)
        raise HTTPException(status_code=503, detail="Fusion-Bench not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Bench trigger failed")
        raise HTTPException(status_code=500, detail=str(e))
