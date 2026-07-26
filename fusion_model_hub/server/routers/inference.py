import logging
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep, SettingsDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])

_LOADED_TTL = 3600
_loaded_models: dict[str, dict] = {}


async def _cleanup_loaded_models() -> None:
    now = time.time()
    expired = [(k, v) for k, v in _loaded_models.items() if now - v.get("loaded_at", 0) > _LOADED_TTL]
    for model_id, info in expired:
        model_name = info.get("model_name", "")
        if model_name:
            try:
                from ..deps import get_settings
                settings = get_settings()
                async with httpx.AsyncClient(timeout=15.0) as client:
                    await client.post(
                        f"{settings.mlx_url}/api/unload",
                        json={"model": model_name},
                    )
            except Exception as e:
                logger.warning("MLX unload during TTL eviction failed for %s: %s", model_id, e)
        _loaded_models.pop(model_id, None)
        logger.info("TTL evicted model: id=%s name=%s", model_id, model_name)


class ServeRequest(BaseModel):
    version_id: str = ""
    gpu: bool = True


@router.post("/models/{model_id}/serve")
async def serve_model(model_id: str, body: ServeRequest, session: SessionDep, settings: SettingsDep):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    version_id = body.version_id
    if not version_id and m.versions:
        published = [v for v in m.versions if v.status.value == "published"]
        if published:
            version_id = published[0].id
        elif m.versions:
            version_id = m.versions[0].id

    if not version_id:
        raise HTTPException(status_code=400, detail="No version available to serve")

    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    model_name = m.hf_repo or m.name
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.mlx_url}/api/load",
                json={"model": model_name, "gpu": body.gpu},
            )
            resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"MLX load failed: {e.response.text}")

    _loaded_models[model_id] = {
        "version_id": version_id,
        "model_name": model_name,
        "status": "loaded",
        "loaded_at": time.time(),
    }
    await _cleanup_loaded_models()
    logger.info("Model served: id=%s version=%s mlx_model=%s", model_id, version_id, model_name)
    return {"model_id": model_id, "version_id": version_id, "status": "loaded", "mlx_model": model_name}


@router.delete("/models/{model_id}/serve")
async def unload_model(model_id: str, settings: SettingsDep):
    if model_id not in _loaded_models:
        raise HTTPException(status_code=404, detail="Model not loaded")

    model_name = _loaded_models[model_id]["model_name"]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{settings.mlx_url}/api/unload",
                json={"model": model_name},
            )
    except Exception as e:
        logger.warning("MLX unload failed: %s", e)

    _loaded_models.pop(model_id, None)
    logger.info("Model unloaded: id=%s", model_id)
    return {"model_id": model_id, "status": "unloaded"}


@router.get("/models/{model_id}/serve")
async def get_serve_status(model_id: str):
    info = _loaded_models.get(model_id)
    if not info:
        return {"model_id": model_id, "status": "not_loaded"}
    return {"model_id": model_id, **info}


@router.post("/inference/{model_id}/chat")
async def chat_completion(model_id: str, body: dict, settings: SettingsDep):
    info = _loaded_models.get(model_id)
    if not info:
        raise HTTPException(status_code=400, detail="Model not loaded — serve it first")

    model_name = info["model_name"]
    payload = {**body, "model": model_name}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{settings.mlx_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@router.post("/inference/{model_id}/completions")
async def text_completion(model_id: str, body: dict, settings: SettingsDep):
    info = _loaded_models.get(model_id)
    if not info:
        raise HTTPException(status_code=400, detail="Model not loaded — serve it first")

    model_name = info["model_name"]
    payload = {**body, "model": model_name}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{settings.mlx_url}/v1/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@router.post("/inference/{model_id}/embeddings")
async def embeddings(model_id: str, body: dict, settings: SettingsDep):
    info = _loaded_models.get(model_id)
    if not info:
        raise HTTPException(status_code=400, detail="Model not loaded — serve it first")

    model_name = info["model_name"]
    payload = {**body, "model": model_name}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.mlx_url}/v1/embeddings", json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
