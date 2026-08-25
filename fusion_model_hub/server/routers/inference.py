import asyncio
import json
import logging
import random
import time

import anyio
import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...db import crud
from ...db.models import ModelStatus
from ..deps import SessionDep, SettingsDep, get_session_factory
from ..errors import safe_http_error
from .models import _check_model_owner, _check_model_read

logger = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])


def _mlx_headers(settings) -> dict[str, str]:
    headers = {"X-Fusion-Source": "model-hub"}
    if settings.mlx_internal_api_key:
        headers["Authorization"] = f"Bearer {settings.mlx_internal_api_key}"
    return headers

_LOADED_TTL = 3600
_loaded_models: dict[str, dict] = {}
_model_stats: dict[str, dict] = {}
_VALID_MODULES = {"chat", "code", "design", "rag", "agent"}
_loaded_lock = asyncio.Lock()
# R5: _cleanup_loaded_models ran after EVERY serve_model, holding _loaded_lock
# for the whole sweep and firing 2 DB queries (pinned + ttl) PER loaded model
# inside the lock — on a hot inference path that serializes every request.
# Throttle to at most once per window so a burst of serves sweeps once, not Nx.
_CLEANUP_MIN_INTERVAL = 60
_last_cleanup_ts: float = 0.0
# R6: _model_stats accumulated forever and was never popped on unload — a long
# run with many transient models leaks unbounded memory exposed via
# /v1/metrics/json. Cap the tracked set; oldest entry evicted when exceeded.
_MODEL_STATS_CAP = 500


def _compute_file_hash(file_path: str) -> str:
    # E-E8: delegate to the shared utils helper so chunk size is consistent.
    from ...utils.hashing import compute_sha256

    return compute_sha256(file_path)


async def _check_module_access(model_id: str, request) -> None:
    module = request.headers.get("X-Fusion-Module", "")
    if not module:
        return
    module = module.lower().strip()
    if module not in _VALID_MODULES:
        return
    try:
        from ..deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            m = await crud.get_model(session, model_id)
            if not m or not m.model_modules:
                return
            allowed = {x.strip() for x in m.model_modules.split(",") if x.strip()}
            if allowed and module not in allowed:
                raise HTTPException(status_code=403, detail=f"Module '{module}' not allowed for this model")
    except HTTPException:
        raise
    except Exception:
        logger.warning("Module access check failed for model=%s", model_id, exc_info=True)


async def _resolve_model_name_for_inference(model_id: str) -> tuple[str, str | None]:
    try:
        from ..deps import get_session_factory
        sf = get_session_factory()
        async with sf() as session:
            deployments = await crud.list_deployments(session, model_id=model_id, status="running")
            for d in deployments:
                if d.gray_enabled and d.gray_version_id and random.randint(1, 100) <= d.gray_traffic_ratio:
                        gray_ver = await crud.get_version(session, d.gray_version_id)
                        if gray_ver:
                            m = await crud.get_model(session, model_id)
                            model_name = m.hf_repo or m.name if m else model_id
                            return model_name, d.gray_version_id
    except Exception:
        logger.warning("Gray route resolution failed for model=%s, using default", model_id, exc_info=True)
    return "", None


async def _get_model_idle_ttl(model_id: str) -> int:
    try:
        sf = get_session_factory()
        async with sf() as session:
            m = await crud.get_model(session, model_id)
            if m:
                if m.ttl_seconds is not None and m.ttl_seconds > 0:
                    return m.ttl_seconds
                return m.idle_timeout_minutes * 60
    except Exception:
        logger.warning("Failed to get model idle timeout for model=%s", model_id, exc_info=True)
    return _LOADED_TTL


async def _is_model_pinned(model_id: str) -> bool:
    try:
        sf = get_session_factory()
        async with sf() as session:
            m = await crud.get_model(session, model_id)
            return bool(m and m.pinned)
    except Exception:
        logger.warning("Failed to check model pinned status for model=%s", model_id, exc_info=True)
        return False


async def _cleanup_loaded_models() -> None:
    # R5: throttle — skip the sweep entirely if one ran inside the window. The
    # non-blocking check avoids serializing on the lock just to decide "skip".
    global _last_cleanup_ts
    now = time.time()
    if now - _last_cleanup_ts < _CLEANUP_MIN_INTERVAL:
        return
    async with _loaded_lock:
        now = time.time()
        expired = []
        for k, v in list(_loaded_models.items()):
            pinned = await _is_model_pinned(k)
            if pinned:
                continue
            model_ttl = await _get_model_idle_ttl(k)
            if now - v.get("loaded_at", 0) > model_ttl:
                expired.append((k, v))
        for model_id, info in expired:
            current = _loaded_models.get(model_id)
            if not current or current is not info:
                continue
            model_name = info.get("model_name", "")
            if model_name:
                try:
                    from ..deps import get_settings
                    settings = get_settings()
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        await client.post(
                            f"{settings.mlx_url}/v1/models/{model_name}/unload",
                            headers=_mlx_headers(settings),
                        )
                except Exception as e:
                    logger.warning("MLX unload during TTL eviction failed for %s: %s", model_id, e)
            _loaded_models.pop(model_id, None)
            _model_stats.pop(model_id, None)
            logger.info("TTL evicted model: id=%s name=%s", model_id, model_name)
        _last_cleanup_ts = time.time()


def _update_model_stats(
    model_id: str, latency_ms: float, tokens: int = 0,
    source_module: str = "", key_id: str = "",
) -> None:
    if model_id not in _model_stats:
        # R6: bound the tracked set so a long run with many transient models
        # cannot grow _model_stats without limit. Evict the oldest-tracked
        # entry (smallest first_request_at) when the cap is reached.
        if len(_model_stats) >= _MODEL_STATS_CAP:
            oldest = min(_model_stats, key=lambda k: _model_stats[k].get("first_request_at", 0))
            _model_stats.pop(oldest, None)
        _model_stats[model_id] = {
            "request_count": 0,
            "total_latency": 0.0,
            "total_tokens": 0,
            "first_request_at": time.time(),
            "last_request_at": None,
            "source_module": "",
            # E-E7: per-key breakdown so /auth/keys/{id}/usage returns ONLY this
            # key's inference volume, not the global aggregate. The top-level
            # counters stay (TTL eviction, /v1/metrics/json still see totals);
            # per_key is the scoping dimension the usage endpoint filters on.
            "per_key": {},
        }
    stats = _model_stats[model_id]
    stats["request_count"] += 1
    stats["total_latency"] += latency_ms
    stats["total_tokens"] += tokens
    stats["last_request_at"] = time.time()
    if source_module:
        stats["source_module"] = source_module
    # E-E7: accumulate per-key counters. Anonymous (no api_key_id, e.g. auth
    # disabled) buckets under "" so local mode still reports a single bucket
    # rather than dropping the volume.
    pk = stats["per_key"].setdefault(key_id or "", {
        "request_count": 0, "total_tokens": 0, "total_latency": 0.0,
    })
    pk["request_count"] += 1
    pk["total_tokens"] += tokens
    pk["total_latency"] += latency_ms


async def _write_inference_audit(model_id: str, action_type: str, latency_ms: float, request: Request) -> None:
    try:
        sf = get_session_factory()
        async with sf() as session:
            module = request.headers.get("X-Fusion-Module", "")
            detail = json.dumps({
                "module": module,
                "model_id": model_id,
                "latency_ms": round(latency_ms, 2),
            })
            await crud.create_audit_log(
                session,
                action=f"inference_{action_type}",
                resource_type="inference",
                resource_id=model_id,
                api_key_id=getattr(request.state, "api_key_id", "") if hasattr(request, "state") else "",
                tenant_id=getattr(request.state, "tenant_id", "") if hasattr(request, "state") else "",
                detail=detail,
            )
    except Exception:
        logger.warning("Failed to write inference audit log for model=%s", model_id, exc_info=True)


class ServeRequest(BaseModel):
    version_id: str = ""
    gpu: bool = True


@router.post("/models/{model_id}/serve")
async def serve_model(model_id: str, body: ServeRequest, session: SessionDep, settings: SettingsDep, request: Request):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    # P0-C: tenant isolation — a non-admin caller may only serve models in their
    # own tenant. Without this, tenant A's key could serve (and thus load on the
    # shared MLX) tenant B's model by id.
    _check_model_owner(m, request)

    if m.model_status == ModelStatus.DRAFT:
        raise HTTPException(status_code=403, detail="Model not published. Only published models can be served.")
    if m.model_status == ModelStatus.DEPRECATED:
        raise HTTPException(status_code=403, detail="Model is deprecated. New serve requests are not allowed.")

    version_id = body.version_id
    if not version_id and m.versions:
        published = [v for v in m.versions if v.status.value == "published"]
        if published:
            # E-D6: the prior code took published[0] with no ordering guarantee —
            # DB insertion order is not guaranteed across updates/repromotes, so
            # the served version was nondeterministic. Serve the most recently
            # created published version (stable, deterministic selection).
            published.sort(key=lambda x: x.created_at or 0, reverse=True)
            version_id = published[0].id
        elif m.versions:
            version_id = m.versions[0].id

    if not version_id:
        raise HTTPException(status_code=400, detail="No version available to serve")

    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")

    if v.file_path:
        import os
        if not os.path.exists(v.file_path):
            # E-D4: a missing file for a PUBLISHED version is a zombie — the
            # prior 403 left it published and silently un-servable. Log loudly
            # with the version id so an operator can retire/rollback it; the
            # 403 still protects callers.
            logger.error(
                "Published version file missing (zombie): version_id=%s path=%s "
                "— retire or rollback this version",
                version_id, v.file_path,
            )
            raise HTTPException(status_code=403, detail="File integrity check failed. Model files may be corrupted.")
        if not v.file_hash:
            computed = await anyio.to_thread.run_sync(_compute_file_hash, v.file_path)
            await crud.update_version(session, version_id, file_hash=computed)
            logger.info("Computed and stored file_hash for version %s", version_id)
        else:
            computed = await anyio.to_thread.run_sync(_compute_file_hash, v.file_path)
            if computed != v.file_hash.lower():
                logger.error(
                    "File hash mismatch for version %s: "
                    "expected=%s computed=%s",
                    version_id, v.file_hash, computed,
                )
                raise HTTPException(
                    status_code=403,
                    detail="File integrity check failed. "
                           "Model files may be corrupted.",
                )

    model_name = m.hf_repo or m.name
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.mlx_url}/v1/models/{model_name}/load",
                json={"gpu": body.gpu},
                headers=_mlx_headers(settings),
            )
            resp.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise safe_http_error(
            e.response.status_code, "Fusion-MLX model load failed",
            exc=e, context="load",
        )

    async with _loaded_lock:
        _loaded_models[model_id] = {
            "version_id": version_id,
            "model_name": model_name,
            "status": "loaded",
            "loaded_at": time.time(),
        }
    await _cleanup_loaded_models()
    logger.info("Model served: id=%s version=%s mlx_model=%s", model_id, version_id, model_name)
    return {
        "model_id": model_id,
        "version_id": version_id,
        "status": "loaded",
        "mlx_model": model_name,
        "pinned": m.pinned,
    }


class HotReloadRequest(BaseModel):
    version_id: str


@router.post("/models/{model_id}/hot-reload")
async def hot_reload_model(
    model_id: str, body: HotReloadRequest, session: SessionDep, settings: SettingsDep, request: Request,
):
    # FR-015 zero-downtime hot reload: preload new version, swap served record,
    # then dispatch webhook. MLX loads by hf_repo; version swap is recorded at
    # hub layer. Real model swap handled by gateway/MLX routing. Hub does
    # preload + bookkeeping + notification.
    info = _loaded_models.get(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Model not currently served; use /serve first")
    old_version_id = info.get("version_id", "")
    if body.version_id == old_version_id:
        raise HTTPException(status_code=400, detail="Target version is already served")
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    # P0-C: tenant isolation on hot-reload (write op).
    _check_model_owner(m, request)
    v = await crud.get_version(session, body.version_id)
    if not v or v.model_id != model_id:
        raise HTTPException(status_code=404, detail="Target version not found for this model")
    if v.status.value != "published":
        raise HTTPException(status_code=403, detail="Only published versions can be hot-reloaded to")
    model_name = m.hf_repo or m.name
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.mlx_url}/v1/models/{model_name}/load",
                json={"gpu": True},
                headers=_mlx_headers(settings),
            )
            if resp.status_code not in (200, 409):
                logger.warning("Hot-reload preload returned %d for %s", resp.status_code, model_name)
                raise HTTPException(status_code=502, detail=f"MLX preload failed: {resp.status_code}")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Hot-reload preload error for %s: %s", model_name, e)
        raise safe_http_error(502, "Fusion-MLX preload failed", exc=e, context="hot-reload-preload")
    async with _loaded_lock:
        _loaded_models[model_id] = {
            "version_id": body.version_id,
            "model_name": model_name,
            "status": "loaded",
            "loaded_at": time.time(),
        }
    logger.info(
        "Hot-reload done: id=%s old_ver=%s new_ver=%s mlx_model=%s",
        model_id, old_version_id, body.version_id, model_name,
    )
    try:
        from .webhooks import dispatch_webhook_event
        await dispatch_webhook_event(
            "model.hot_reloaded",
            {
                "model_id": model_id,
                "model_name": model_name,
                "old_version_id": old_version_id,
                "new_version_id": body.version_id,
            },
            tenant_id=getattr(m, "tenant_id", "") or "",
        )
    except Exception as e:
        logger.warning("Hot-reload webhook dispatch failed: %s", e)
    return {
        "model_id": model_id,
        "old_version_id": old_version_id,
        "new_version_id": body.version_id,
        "status": "reloaded",
        "mlx_model": model_name,
    }


@router.delete("/models/{model_id}/serve")
async def unload_model(model_id: str, settings: SettingsDep, session: SessionDep, request: Request):
    async with _loaded_lock:
        if model_id not in _loaded_models:
            raise HTTPException(status_code=404, detail="Model not loaded")
        model_name = _loaded_models[model_id]["model_name"]

    # P0-C: tenant isolation — only the model's tenant (or admin) may unload it.
    # A served model always has a DB row, so a miss is the test-injected loaded
    # shortcut; production rows are checked and ownership enforced.
    m = await crud.get_model(session, model_id)
    if m:
        _check_model_owner(m, request)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{settings.mlx_url}/v1/models/{model_name}/unload",
                headers=_mlx_headers(settings),
            )
    except Exception as e:
        logger.warning("MLX unload failed: %s", e)

    async with _loaded_lock:
        _loaded_models.pop(model_id, None)
    _model_stats.pop(model_id, None)
    logger.info("Model unloaded: id=%s", model_id)
    return {"model_id": model_id, "status": "unloaded"}


@router.get("/models/{model_id}/serve")
async def get_serve_status(model_id: str, session: SessionDep, request: Request):
    # P0-C: tenant isolation on read — a key scoped to tenant A must not see
    # tenant B's serve status (which leaks the served version + mlx model name).
    # A non-existent id returns not_loaded (200) to avoid leaking existence; a
    # real served row is checked and isolation enforced.
    m = await crud.get_model(session, model_id)
    if m:
        _check_model_read(m, request)
    info = _loaded_models.get(model_id)
    if not info:
        return {"model_id": model_id, "status": "not_loaded"}
    return {"model_id": model_id, **info}


@router.post("/models/{model_id}/pin")
async def pin_model(model_id: str, session: SessionDep, request: Request):
    existing = await crud.get_model(session, model_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Model not found")
    # P0-C: tenant isolation — pin is a write; only the owner tenant / admin.
    _check_model_owner(existing, request)
    await crud.update_model(session, model_id, pinned=True)
    logger.info("Model pinned: id=%s", model_id)
    return {"model_id": model_id, "pinned": True}


@router.delete("/models/{model_id}/pin")
async def unpin_model(model_id: str, session: SessionDep, request: Request):
    existing = await crud.get_model(session, model_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Model not found")
    # P0-C: tenant isolation on unpin (write).
    _check_model_owner(existing, request)
    await crud.update_model(session, model_id, pinned=False)
    logger.info("Model unpinned: id=%s", model_id)
    return {"model_id": model_id, "pinned": False}


@router.post("/inference/{model_id}/chat")
async def chat_completion(model_id: str, body: dict, settings: SettingsDep, request: Request, session: SessionDep):
    await _check_module_access(model_id, request)
    # P0-C: tenant isolation — a key scoped to tenant A must not run inference
    # against tenant B's model by id (cross-tenant read -> 404, no existence leak).
    # A served model always has a DB row (serve_model creates-then-loads), so the
    # get_model miss path is only the test-injected _loaded_models shortcut; in
    # production the row exists and the read check enforces isolation.
    m = await crud.get_model(session, model_id)
    if m:
        _check_model_read(m, request)
    info = _loaded_models.get(model_id)
    if not info:
        raise HTTPException(status_code=400, detail="Model not loaded — serve it first")

    model_name = info["model_name"]
    _, gray_ver = await _resolve_model_name_for_inference(model_id)
    if gray_ver:
        try:
            sf = __import__("fusion_model_hub.server.deps", fromlist=["get_session_factory"]).get_session_factory()
            async with sf() as s:
                gv = await crud.get_version(s, gray_ver)
                if gv:
                    gm = await crud.get_model(s, gv.model_id)
                    if gm:
                        model_name = gm.hf_repo or gm.name
        except Exception:
            logger.warning("Gray version model lookup failed for model=%s", model_id, exc_info=True)

    payload = {**body, "model": model_name}
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{settings.mlx_url}/v1/chat/completions",
                json=payload,
                headers=_mlx_headers(settings),
            )
            resp.raise_for_status()
            result = resp.json()
        latency_ms = (time.time() - start_time) * 1000
        tokens = 0
        usage = result.get("usage", {})
        if usage:
            tokens = usage.get("total_tokens", 0)
        _update_model_stats(
            model_id, latency_ms, tokens,
            request.headers.get("X-Fusion-Module", "").lower(),
            key_id=getattr(request.state, "api_key_id", ""),
        )
        await _write_inference_audit(model_id, "chat", latency_ms, request)
        return result
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise safe_http_error(
            e.response.status_code, "Fusion-MLX chat request failed",
            exc=e, context="chat-completions",
        )


@router.post("/inference/{model_id}/completions")
async def text_completion(model_id: str, body: dict, settings: SettingsDep, request: Request, session: SessionDep):
    await _check_module_access(model_id, request)
    # P0-C: tenant isolation on inference read (cross-tenant -> 404). A served
    # model always has a DB row, so a miss is only the test-injected loaded
    # shortcut; production rows are checked and isolation enforced.
    m = await crud.get_model(session, model_id)
    if m:
        _check_model_read(m, request)
    info = _loaded_models.get(model_id)
    if not info:
        raise HTTPException(status_code=400, detail="Model not loaded — serve it first")

    model_name = info["model_name"]
    _, gray_ver = await _resolve_model_name_for_inference(model_id)
    if gray_ver:
        try:
            sf = __import__("fusion_model_hub.server.deps", fromlist=["get_session_factory"]).get_session_factory()
            async with sf() as s:
                gv = await crud.get_version(s, gray_ver)
                if gv:
                    gm = await crud.get_model(s, gv.model_id)
                    if gm:
                        model_name = gm.hf_repo or gm.name
        except Exception:
            logger.warning("Gray version model lookup failed for model=%s", model_id, exc_info=True)

    payload = {**body, "model": model_name}
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{settings.mlx_url}/v1/completions", json=payload, headers=_mlx_headers(settings))
            resp.raise_for_status()
            result = resp.json()
        latency_ms = (time.time() - start_time) * 1000
        tokens = 0
        usage = result.get("usage", {})
        if usage:
            tokens = usage.get("total_tokens", 0)
        _update_model_stats(
            model_id, latency_ms, tokens,
            request.headers.get("X-Fusion-Module", "").lower(),
            key_id=getattr(request.state, "api_key_id", ""),
        )
        await _write_inference_audit(model_id, "completions", latency_ms, request)
        return result
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise safe_http_error(
            e.response.status_code, "Fusion-MLX completions request failed",
            exc=e, context="completions",
        )


@router.post("/inference/{model_id}/embeddings")
async def embeddings(model_id: str, body: dict, settings: SettingsDep, request: Request, session: SessionDep):
    await _check_module_access(model_id, request)
    # P0-C: tenant isolation on inference read (cross-tenant -> 404). A served
    # model always has a DB row, so a miss is only the test-injected loaded
    # shortcut; production rows are checked and isolation enforced.
    m = await crud.get_model(session, model_id)
    if m:
        _check_model_read(m, request)
    info = _loaded_models.get(model_id)
    if not info:
        raise HTTPException(status_code=400, detail="Model not loaded — serve it first")

    model_name = info["model_name"]
    payload = {**body, "model": model_name}
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.mlx_url}/v1/embeddings", json=payload, headers=_mlx_headers(settings))
            resp.raise_for_status()
            result = resp.json()
        latency_ms = (time.time() - start_time) * 1000
        tokens = 0
        usage = result.get("usage", {})
        if usage:
            tokens = usage.get("total_tokens", 0)
        _update_model_stats(
            model_id, latency_ms, tokens,
            request.headers.get("X-Fusion-Module", "").lower(),
            key_id=getattr(request.state, "api_key_id", ""),
        )
        await _write_inference_audit(model_id, "embeddings", latency_ms, request)
        return result
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Fusion-MLX server unavailable")
    except httpx.HTTPStatusError as e:
        raise safe_http_error(
            e.response.status_code, "Fusion-MLX embeddings request failed",
            exc=e, context="embeddings",
        )
