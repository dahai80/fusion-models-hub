import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ...db.models import (
    ModelStatus,
    ModelType,
)
from ..deps import CacheDep, SessionDep, StoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])

HF_MIRROR = "https://hf-mirror.com"


def _check_model_owner(model, request: Request):
    # E-S9: the prior check compared model.owner (a forgeable free-text field any
    # caller sets at create) against the API-key *name* (not its tenant/id) — so
    # two unrelated keys with the same name passed, and a renamed owner blocked
    # the real owner. Enforce tenant isolation instead: a non-admin caller may
    # only modify models in their own tenant; admin passes through. When auth is
    # off (local dev) the check is permissive, matching list_models semantics.
    from ..auth import _is_auth_enabled

    if not _is_auth_enabled():
        return
    role = getattr(request.state, "user_role", "")
    if role == "admin":
        return
    caller_tenant = getattr(request.state, "tenant_id", "") or ""
    model_tenant = getattr(model, "tenant_id", "") or ""
    if caller_tenant and model_tenant and caller_tenant != model_tenant:
        raise HTTPException(status_code=403, detail="Model belongs to a different tenant")


def _check_model_read(model, request: Request):
    # E-S9: GET /models/{id} previously had no tenant guard — a key scoped to
    # tenant A could read tenant B's model by id. Mirrors _check_model_owner but
    # for reads: cross-tenant reads 404 (not 403, to avoid leaking existence).
    from ..auth import _is_auth_enabled

    if not _is_auth_enabled():
        return
    role = getattr(request.state, "user_role", "")
    if role == "admin":
        return
    caller_tenant = getattr(request.state, "tenant_id", "") or ""
    model_tenant = getattr(model, "tenant_id", "") or ""
    if caller_tenant and model_tenant and caller_tenant != model_tenant:
        raise HTTPException(status_code=404, detail="Model not found")


class ModelCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=64)
    description: str = ""
    model_type: ModelType = ModelType.LLM
    base_model_id: str = ""
    architecture: str = ""
    params_size: str = ""
    license: str = ""
    author: str = ""
    language: str = ""
    task_types: str = ""
    owner: str = ""
    hf_repo: str = ""
    model_modules: str = ""
    idle_timeout_minutes: int = 60
    tags: list[dict[str, str]] = Field(default_factory=list)


class ModelUpdate(BaseModel):
    description: str | None = None
    model_type: ModelType | None = None
    base_model_id: str | None = None
    architecture: str | None = None
    params_size: str | None = None
    license: str | None = None
    author: str | None = None
    language: str | None = None
    task_types: str | None = None
    owner: str | None = None
    hf_repo: str | None = None
    model_modules: str | None = None
    idle_timeout_minutes: int | None = None
    ttl_seconds: int | None = None
    model_status: str | None = None
    tags: list[dict[str, str]] | None = None


class SyncRequest(BaseModel):
    source_url: str = Field(..., description="Remote Fusion Model Hub base URL")
    dry_run: bool = False


class BatchDeleteRequest(BaseModel):
    model_ids: list[str]


class BatchTagRequest(BaseModel):
    model_ids: list[str]
    tags: list[dict[str, str]]


def _model_to_dict(m) -> dict[str, Any]:
    return {
        "id": m.id,
        "name": m.name,
        "tenant_id": m.tenant_id,
        "description": m.description,
        "model_type": m.model_type.value,
        "base_model_id": m.base_model_id,
        "architecture": m.architecture,
        "params_size": m.params_size,
        "license": m.license,
        "author": m.author,
        "language": m.language,
        "task_types": m.task_types,
        "owner": m.owner,
        "hf_repo": m.hf_repo,
        "download_count": m.download_count,
        "model_modules": m.model_modules,
        "idle_timeout_minutes": m.idle_timeout_minutes,
        "ttl_seconds": m.ttl_seconds,
        "model_status": m.model_status.value,
        "tags": [{"key": t.key, "value": t.value} for t in m.tags],
        "versions_count": len(m.versions),
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }


# -- Static-path routes MUST come before /models/{model_id} --


@router.post("/models", status_code=201)
async def create_model(body: ModelCreate, session: SessionDep, request: Request):
    existing = await crud.get_model_by_name(session, body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Model name already exists: {body.name}")
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    if body.model_type == ModelType.LORA:
        if not body.base_model_id:
            raise HTTPException(
                status_code=400,
                detail="base_model_id is required for LoRA adapter models",
            )
        base_m = await crud.get_model(session, body.base_model_id)
        if not base_m:
            raise HTTPException(status_code=404, detail="base_model_id not found")
    m = await crud.create_model(
        session,
        name=body.name,
        tenant_id=tenant_id,
        description=body.description,
        model_type=body.model_type,
        base_model_id=body.base_model_id,
        architecture=body.architecture,
        params_size=body.params_size,
        license=body.license,
        author=body.author,
        language=body.language,
        task_types=body.task_types,
        owner=body.owner,
        hf_repo=body.hf_repo,
        model_modules=body.model_modules,
        idle_timeout_minutes=body.idle_timeout_minutes,
    )
    if body.tags:
        await crud.set_tags(session, m.id, body.tags)
        await session.refresh(m)
    try:
        from .webhooks import dispatch_webhook_event

        await dispatch_webhook_event("model.created", {"id": m.id, "name": m.name}, tenant_id=tenant_id)
        if body.model_type == ModelType.LORA:
            await dispatch_webhook_event(
                "adapter.published",
                {"id": m.id, "name": m.name, "base_model_id": body.base_model_id},
                tenant_id=tenant_id,
            )
    except Exception:
        logger.exception("Webhook dispatch failed for model.created")
    return _model_to_dict(m)


@router.get("/models")
async def list_models(
    session: SessionDep,
    request: Request,
    keyword: str = "",
    model_type: str = "",
    architecture: str = "",
    page: int = 1,
    page_size: int = 20,
):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    models, total = await crud.list_models(
        session,
        tenant_id=tenant_id,
        keyword=keyword,
        model_type=model_type,
        architecture=architecture,
        page=page,
        page_size=page_size,
    )
    return {
        "items": [_model_to_dict(m) for m in models],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/models/search")
async def search_models(
    session: SessionDep,
    request: Request,
    keyword: str = "",
    model_type: str = "",
    architecture: str = "",
    params_size: str = "",
    quantization: str = "",
    min_benchmark_score: float = 0.0,
    sort_by: str = "updated_at",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 20,
):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    models, total = await crud.list_models(
        session,
        tenant_id=tenant_id,
        keyword=keyword,
        model_type=model_type,
        architecture=architecture,
        page=1,
        page_size=1000,
    )
    filtered = models
    if params_size:
        filtered = [m for m in filtered if m.params_size == params_size]
    if quantization:
        filtered = [m for m in filtered if any(v.quantization.value == quantization for v in m.versions)]
    if min_benchmark_score > 0:
        filtered = [m for m in filtered if any(v.benchmark_score >= min_benchmark_score for v in m.versions)]
    sort_key_map = {
        "updated_at": lambda m: m.updated_at or m.created_at,
        "created_at": lambda m: m.created_at,
        "download_count": lambda m: m.download_count,
        "name": lambda m: m.name,
        "benchmark_score": lambda m: max((v.benchmark_score for v in m.versions), default=0),
    }
    key_func = sort_key_map.get(sort_by, sort_key_map["updated_at"])
    filtered.sort(key=key_func, reverse=(sort_order == "desc"))
    total = len(filtered)
    offset = (page - 1) * page_size
    page_items = filtered[offset : offset + page_size]
    return {
        "items": [_model_to_dict(m) for m in page_items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/models/market/search")
async def market_search(
    request: Request,
    keyword: str = "",
    source: str = "all",
    task: str = "",
    page: int = 1,
    page_size: int = 20,
):
    results = {"local": [], "huggingface": [], "modelscope": [], "private": []}
    if source in ("all", "local"):
        try:
            from ..deps import get_session_factory

            sf = get_session_factory()
            async with sf() as session:
                tenant_id = getattr(request.state, "tenant_id", "") or ""
                models, _ = await crud.list_models(
                    session,
                    tenant_id=tenant_id,
                    keyword=keyword,
                    page=page,
                    page_size=page_size,
                )
                results["local"] = [_model_to_dict(m) for m in models]
        except Exception:
            logger.exception("Local market search failed")
    if source in ("all", "modelscope"):
        try:
            from ...repo.modelscope_search import search_modelscope

            ms = await search_modelscope(query=keyword, task=task, page=page, page_size=page_size)
            results["modelscope"] = ms.get("items", [])
        except Exception:
            logger.exception("ModelScope search failed")
    if source in ("all", "private"):
        try:
            from ..deps import get_session_factory

            sf = get_session_factory()
            async with sf() as session:
                models, _ = await crud.list_models(
                    session,
                    keyword=keyword,
                    page=page,
                    page_size=page_size,
                )
                private_models = [_model_to_dict(m) for m in models if not m.hf_repo]
                results["private"] = private_models
        except Exception:
            logger.exception("Private repo search failed")
    if source in ("all", "huggingface"):
        try:
            hf_url = "https://hf-mirror.com/api/models"
            params = {"search": keyword, "direction": -1, "sort": "downloads", "limit": page_size}
            if task:
                params["filter"] = task
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(hf_url, params=params)
                if resp.status_code == 200:
                    items = []
                    for m in resp.json():
                        items.append(
                            {
                                "name": m.get("id", ""),
                                "id": m.get("id", ""),
                                "task": m.get("pipeline_tag", ""),
                                "downloads": m.get("downloads", 0),
                                "source": "huggingface",
                            }
                        )
                    results["huggingface"] = items
        except Exception:
            logger.exception("HF market search failed")
    return results


@router.get("/models/compare")
async def compare_models(ids: str, session: SessionDep):
    model_ids = [i.strip() for i in ids.split(",") if i.strip()]
    if len(model_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 model IDs required (comma-separated)")
    results = []
    for mid in model_ids:
        m = await crud.get_model(session, mid)
        if not m:
            raise HTTPException(status_code=404, detail=f"Model not found: {mid}")
        results.append(_model_to_dict(m))
    return {"models": results}


def _validate_url(url_str: str) -> None:
    from ..ssrf import validate_external_url

    validate_external_url(url_str)


@router.post("/models/sync")
async def sync_registry(body: SyncRequest, session: SessionDep):
    logger.info("Registry sync from %s (dry_run=%s)", body.source_url, body.dry_run)
    _validate_url(body.source_url)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{body.source_url}/api/v1/models?page_size=100")
            resp.raise_for_status()
            remote_data = resp.json()
    except httpx.HTTPError as e:
        logger.error("Sync fetch failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Failed to fetch remote registry: {e}")

    remote_items = remote_data.get("items", [])
    local_models, _ = await crud.list_models(session, page=1, page_size=1000)
    local_names = {m.name for m in local_models}

    new_models = [item for item in remote_items if item["name"] not in local_names]
    logger.info("Sync found %d remote models, %d new", len(remote_items), len(new_models))

    if body.dry_run:
        return {"dry_run": True, "new_count": len(new_models), "new_models": [m["name"] for m in new_models]}

    created = []
    for item in new_models:
        m = await crud.create_model(
            session,
            name=item["name"],
            description=item.get("description", ""),
            model_type=ModelType(item["model_type"]) if item.get("model_type") else ModelType.LLM,
            architecture=item.get("architecture", ""),
            params_size=item.get("params_size", ""),
            license=item.get("license", ""),
            author=item.get("author", ""),
            language=item.get("language", ""),
            task_types=item.get("task_types", ""),
            owner=item.get("owner", ""),
            hf_repo=item.get("hf_repo", ""),
        )
        created.append({"id": m.id, "name": m.name})

    return {"synced": len(created), "new_models": created}


@router.post("/models/batch/delete")
async def batch_delete(
    body: BatchDeleteRequest,
    session: SessionDep,
    store: StoreDep,
    cache: CacheDep,
):
    logger.info("Batch delete: %d models", len(body.model_ids))
    deleted = []
    for mid in body.model_ids:
        m = await crud.get_model(session, mid)
        if m:
            # E-R2: store.delete_model_files only rmtree's the models_dir/model_id
            # tree; the 3-level cache lives in a separate cache_dir tree and is
            # never touched, so orphan cache files + dangling index.json entries
            # accumulated unbounded. Drop the cache entries for the model too.
            try:
                cache.remove_model(mid)
            except Exception:
                logger.exception("Cache remove_model failed for %s", mid)
            store.delete_model_files(mid)
            await crud.delete_model(session, mid)
            deleted.append(mid)
    return {"deleted": deleted, "count": len(deleted)}


@router.post("/models/batch/tag")
async def batch_tag(body: BatchTagRequest, session: SessionDep):
    logger.info("Batch tag: %d models", len(body.model_ids))
    updated = []
    for mid in body.model_ids:
        m = await crud.get_model(session, mid)
        if m:
            await crud.set_tags(session, mid, body.tags)
            updated.append(mid)
    return {"tagged": updated, "count": len(updated)}


class HfImportRequest(BaseModel):
    hf_repo: str = Field("", max_length=200)
    name: str = Field("", max_length=128)
    download: bool = False


@router.post("/models/import/hf", status_code=201)
async def import_from_hf(body: HfImportRequest, session: SessionDep):
    hf_repo = body.hf_repo.strip()
    if not hf_repo:
        raise HTTPException(status_code=400, detail="hf_repo is required")
    name = body.name.strip() or hf_repo.split("/")[-1].lower()
    existing = await crud.get_model_by_name(session, name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Model already exists: {name}")

    hf_info = await _fetch_hf_model_info(hf_repo)

    model_type = ModelType.LLM
    pipeline_tag = hf_info.get("pipeline_tag", "")
    type_map = {
        "text-generation": ModelType.LLM,
        "text2text-generation": ModelType.LLM,
        "conversational": ModelType.CHAT,
        "feature-extraction": ModelType.EMBEDDING,
        "image-text-to-text": ModelType.MULTIMODAL,
        "text-to-image": ModelType.IMAGE,
        "automatic-speech-recognition": ModelType.AUDIO,
        "text-to-speech": ModelType.AUDIO,
        "code-generation": ModelType.CODE,
    }
    if pipeline_tag in type_map:
        model_type = type_map[pipeline_tag]

    architecture = ""
    config_data = hf_info.get("config", {}) or {}
    if isinstance(config_data, dict):
        arch_list = config_data.get("architectures", [])
        if arch_list:
            architecture = arch_list[0]

    m = await crud.create_model(
        session,
        name=name,
        description=hf_info.get("description", hf_info.get("cardData", {}).get("description", "")),
        model_type=model_type,
        architecture=architecture,
        params_size=(
            hf_info.get("safetensors", {}).get("total", "") if isinstance(hf_info.get("safetensors"), dict) else ""
        ),
        license=hf_info.get("cardData", {}).get("license", ""),
        author=hf_info.get("author", ""),
        language=",".join(hf_info.get("cardData", {}).get("language", [])),
        task_types=pipeline_tag,
        owner=hf_repo.split("/")[0] if "/" in hf_repo else "",
        hf_repo=hf_repo,
    )
    logger.info("Imported HF model: repo=%s -> id=%s type=%s", hf_repo, m.id, model_type.value)

    if body.download:
        from ...repo.downloader import ModelDownloader

        download_url = f"{HF_MIRROR}/{hf_repo}"
        downloader = ModelDownloader()
        result = await downloader.download(download_url, name)
        if result.get("status") == "completed":
            try:
                version = await crud.create_version(
                    session,
                    model_id=m.id,
                    version="hf-default",
                    file_path=result["path"],
                    file_size=result.get("size_bytes", 0),
                )
            except crud.VersionConflictError:
                # P1-D: a concurrent import already created hf-default for this
                # new model — rare (new id each import) but guard against 500.
                version = None
                logger.warning("HF import: hf-default already exists for model=%s", m.id)
            if version:
                logger.info(
                    "Downloaded HF model files: repo=%s version_id=%s path=%s",
                    hf_repo,
                    version.id,
                    result["path"],
                )
        else:
            logger.warning("HF model download failed: repo=%s error=%s", hf_repo, result.get("error", "unknown"))

    return _model_to_dict(m)


async def _fetch_hf_model_info(repo_id: str) -> dict:
    url = f"{HF_MIRROR}/api/models/{repo_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning("HF API fetch failed for %s: %s", repo_id, e)
        return {}


# -- Dynamic path routes AFTER all static paths --


@router.get("/models/{model_id}")
async def get_model(model_id: str, session: SessionDep, request: Request):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    _check_model_read(m, request)
    data = _model_to_dict(m)
    data["versions"] = [
        {
            "id": v.id,
            "version": v.version,
            "format": v.format.value,
            "quantization": v.quantization.value,
            "status": v.status.value,
            "file_size": v.file_size,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in m.versions
    ]
    return data


@router.put("/models/{model_id}")
async def update_model(model_id: str, body: ModelUpdate, session: SessionDep, request: Request):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    _check_model_owner(m, request)
    fields = body.model_dump(exclude_unset=True, exclude={"tags"})
    if fields:
        m = await crud.update_model(session, model_id, **fields)
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
    if body.tags is not None:
        await crud.set_tags(session, model_id, body.tags)
        await session.refresh(m)
    return _model_to_dict(m)


class ModelModulesUpdate(BaseModel):
    modules: str = Field(..., description="Comma-separated Fusion modules: chat,code,design,rag,agent")


@router.put("/models/{model_id}/modules")
async def update_model_modules(model_id: str, body: ModelModulesUpdate, session: SessionDep, request: Request):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    _check_model_owner(m, request)
    m = await crud.update_model(session, model_id, model_modules=body.modules)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    logger.info("Updated model modules: id=%s modules=%s", model_id, body.modules)
    return {"id": m.id, "model_modules": m.model_modules}


@router.post("/models/{model_id}/publish")
async def publish_model(model_id: str, session: SessionDep, request: Request):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    _check_model_owner(m, request)
    if hasattr(request.state, "user_role") and request.state.user_role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can publish models")
    m = await crud.update_model(session, model_id, model_status=ModelStatus.PUBLISHED)
    logger.info("Model published: id=%s", model_id)
    return {"id": model_id, "model_status": "published"}


@router.post("/models/{model_id}/deprecate")
async def deprecate_model(model_id: str, session: SessionDep, request: Request):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    _check_model_owner(m, request)
    if hasattr(request.state, "user_role") and request.state.user_role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can deprecate models")
    m = await crud.update_model(session, model_id, model_status=ModelStatus.DEPRECATED)
    logger.info("Model deprecated: id=%s", model_id)
    return {"id": model_id, "model_status": "deprecated"}


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    session: SessionDep,
    store: StoreDep,
    request: Request,
    cache: CacheDep,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    _check_model_owner(m, request)
    # E-R2: the cache tree is separate from store.delete_model_files; purge it
    # too or the model's quantized/converted cache files outlive the DB row.
    try:
        cache.remove_model(model_id)
    except Exception:
        logger.exception("Cache remove_model failed for %s", model_id)
    store.delete_model_files(model_id)
    tenant_id = m.tenant_id
    await crud.delete_model(session, model_id)
    try:
        from .webhooks import dispatch_webhook_event

        await dispatch_webhook_event("model.deleted", {"id": model_id, "name": m.name}, tenant_id=tenant_id)
    except Exception:
        logger.exception("Webhook dispatch failed for model.deleted")
    return {"status": "deleted", "id": model_id}
