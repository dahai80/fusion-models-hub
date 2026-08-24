import asyncio
import hashlib
import json
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ...db import crud
from ...db.crud import EvaluationThresholdError, InvalidTransition
from ...db.models import ModelFormat, Quantization, VersionStatus
from ...repo.downloader import ModelDownloader
from ..deps import SessionDep, SettingsDep, StoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["versions"])

_running_url_downloads: dict[str, asyncio.Task] = {}


class VersionCreate(BaseModel):
    version: str
    format: ModelFormat = ModelFormat.MLX
    quantization: Quantization = Quantization.Q4
    release_notes: str = ""


class StatusChange(BaseModel):
    target_status: VersionStatus
    remark: str = ""
    approval_level: str = "l1"


class BenchmarkResult(BaseModel):
    benchmark_score: float | None = None
    inference_latency: float | None = None
    throughput: float | None = None
    memory_usage: float | None = None
    context_length: int | None = None


def _version_to_dict(v) -> dict[str, Any]:
    return {
        "id": v.id,
        "model_id": v.model_id,
        "version": v.version,
        "format": v.format.value,
        "quantization": v.quantization.value,
        "status": v.status.value,
        "file_path": v.file_path,
        "file_hash": v.file_hash,
        "file_size": v.file_size,
        "release_notes": v.release_notes,
        "benchmark_score": v.benchmark_score,
        "inference_latency": v.inference_latency,
        "throughput": v.throughput,
        "memory_usage": v.memory_usage,
        "context_length": v.context_length,
        "successor_version_id": v.successor_version_id,
        "encrypted": v.encrypted,
        "license_type": v.license_type,
        "data_compliance": v.data_compliance,
        "created_at": v.created_at.isoformat() if v.created_at else None,
    }


MAX_TOTAL_CHUNKS = 10000
CHUNK_READ_SIZE = 5 * 1024 * 1024


def _caller_tenant(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or ""


async def _enforce_version_tenant(session, v, request: Request) -> None:
    # F-04: cross-tenant read/download/state-change guard. Empty caller tenant
    # (auth-disabled local mode) is permissive — matches list_models semantics.
    tenant_id = _caller_tenant(request)
    if not tenant_id or v is None:
        return
    m = await crud.get_model(session, v.model_id)
    if not m or m.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Version not found")


def _chunk_upload_token(model_id: str, version: str) -> str:
    # Deterministic per (model_id, version) so a multi-chunk upload session
    # resolves to one upload dir across requests, yet is opaque to callers
    # (prevents path-guessing collisions between concurrent uploads).
    raw = f"{model_id}:{version}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


@router.post("/models/{model_id}/versions", status_code=201)
async def upload_version(
    model_id: str,
    version: str = Form(""),
    format: ModelFormat = Form(ModelFormat.MLX),
    quantization: Quantization = Form(Quantization.Q4),
    release_notes: str = Form(""),
    file: UploadFile | None = None,
    session: SessionDep = None,
    store: StoreDep = None,
    settings: SettingsDep = None,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    if not version:
        raise HTTPException(status_code=400, detail="version is required")

    target_dir = store.model_version_dir(model_id, version)
    file_path = ""
    file_hash = ""
    file_size = 0

    if file:
        # F-03: stream the upload and enforce max_upload_size_mb instead of
        # reading the whole body into memory (OOM on oversized uploads).
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        hasher = hashlib.sha256()
        written = 0
        safe_name = os.path.basename(file.filename or "model.bin") or "model.bin"
        target_path = target_dir / safe_name
        with open(target_path, "wb") as out:
            while True:
                block = await file.read(CHUNK_READ_SIZE)
                if not block:
                    break
                written += len(block)
                if written > max_bytes:
                    out.close()
                    try:
                        target_path.unlink()
                    except OSError:
                        pass
                    raise HTTPException(status_code=413, detail="Upload exceeds max_upload_size_mb")
                out.write(block)
                hasher.update(block)
        file_hash = hasher.hexdigest()
        file_size = written
        file_path = str(target_path)
        logger.info("Uploaded file for model=%s version=%s size=%d", model_id, version, file_size)

    v = await crud.create_version(
        session,
        model_id=model_id, version=version, format=format,
        quantization=quantization, file_path=file_path,
        file_hash=file_hash, file_size=file_size,
        release_notes=release_notes,
    )
    if not v:
        raise HTTPException(status_code=500, detail="Failed to create version")
    return _version_to_dict(v)


@router.post("/models/{model_id}/versions/chunk-upload", status_code=201)
async def chunk_upload_version(
    model_id: str,
    session: SessionDep,
    store: StoreDep,
    settings: SettingsDep,
    version: str = Form(""),
    format: ModelFormat = Form(ModelFormat.MLX),
    quantization: Quantization = Form(Quantization.Q4),
    release_notes: str = Form(""),
    filename: str = Form("model.mlx"),
    total_chunks: int = Form(0),
    chunk_index: int = Form(0),
    chunk: UploadFile | None = None,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    # F-03: cap total_chunks, validate chunk_index range and continuity.
    if total_chunks <= 0 or total_chunks > MAX_TOTAL_CHUNKS:
        raise HTTPException(
            status_code=400,
            detail=f"total_chunks must be in 1..{MAX_TOTAL_CHUNKS}",
        )
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="chunk_index out of range")

    # upload_id random token appended so concurrent uploads of the same
    # (model_id, version) don't clobber each other's chunk slots.
    upload_id = f"{model_id}_{version}_{_chunk_upload_token(model_id, version)}"
    if not chunk:
        raise HTTPException(status_code=400, detail="chunk file is required")

    data = await chunk.read()
    await store.write_chunk(upload_id, chunk_index, data)

    if chunk_index < total_chunks - 1:
        return {"status": "chunk_received", "upload_id": upload_id, "chunk_index": chunk_index}

    try:
        target_dir = store.model_version_dir(model_id, version)
        path, hash_val, size = await store.assemble_chunks(upload_id, target_dir, filename, total_chunks)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # F-03: enforce final assembled size against the global cap.
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size > max_bytes:
        logger.warning("Assembled upload exceeds cap: upload=%s size=%d", upload_id, size)
        raise HTTPException(status_code=413, detail="Assembled upload exceeds max_upload_size_mb")

    v = await crud.create_version(
        session,
        model_id=model_id, version=version, format=format,
        quantization=quantization, file_path=str(path),
        file_hash=hash_val, file_size=size,
        release_notes=release_notes,
    )
    if not v:
        raise HTTPException(status_code=500, detail="Failed to create version")
    return _version_to_dict(v)


@router.get("/models/{model_id}/versions")
async def list_versions(
    model_id: str,
    session: SessionDep,
    request: Request,
    status: str = "",
    page: int = 1,
    page_size: int = 20,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    versions, total = await crud.list_versions(
        session, model_id, tenant_id=tenant_id, status=status, page=page, page_size=page_size,
    )
    return {
        "items": [_version_to_dict(v) for v in versions],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/versions/{version_id}")
async def get_version(version_id: str, session: SessionDep, request: Request):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    await _enforce_version_tenant(session, v, request)
    return _version_to_dict(v)


@router.put("/versions/{version_id}/status")
async def update_version_status(version_id: str, body: StatusChange, session: SessionDep, request: Request):
    existing = await crud.get_version(session, version_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Version not found")
    await _enforce_version_tenant(session, existing, request)
    try:
        v = await crud.update_version_status(
            session, version_id, body.target_status,
            approval_level=body.approval_level,
        )
    except InvalidTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
    except EvaluationThresholdError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return _version_to_dict(v)


@router.get("/versions/{version_id}/download")
async def download_version(version_id: str, session: SessionDep, store: StoreDep, request: Request):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    await _enforce_version_tenant(session, v, request)
    if not v.file_path:
        raise HTTPException(status_code=404, detail="No file associated with this version")

    file_path = store.get_file(v.file_path)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found on disk")

    await crud.increment_download(session, v.model_id)

    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@router.put("/versions/{version_id}/benchmark")
async def update_benchmark(version_id: str, body: BenchmarkResult, session: SessionDep):
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No benchmark fields provided")
    v = await crud.update_version(session, version_id, **fields)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return _version_to_dict(v)


class MetricsUpdate(BaseModel):
    inference_latency: float | None = None
    throughput: float | None = None
    memory_usage: float | None = None
    benchmark_score: float | None = None
    context_length: int | None = None


@router.put("/versions/{version_id}/metrics")
async def update_version_metrics(version_id: str, body: MetricsUpdate, session: SessionDep):
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No metrics fields provided")
    v = await crud.update_version(session, version_id, **fields)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    logger.info("Version metrics updated: id=%s fields=%s", version_id, list(fields.keys()))
    return _version_to_dict(v)


@router.post("/versions/{version_id}/rollback")
async def rollback_version(version_id: str, session: SessionDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    try:
        v = await crud.update_version_status(session, version_id, VersionStatus.PUBLISHED)
    except InvalidTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
    try:
        from .webhooks import dispatch_webhook_event
        await dispatch_webhook_event("version.published", {"id": version_id, "model_id": v.model_id})
    except Exception:
        logger.exception("Webhook dispatch failed for version.published")
    return _version_to_dict(v)


class DeprecateRequest(BaseModel):
    successor_version_id: str = ""
    remark: str = ""


@router.post("/versions/{version_id}/deprecate")
async def deprecate_version(version_id: str, body: DeprecateRequest, session: SessionDep):
    try:
        v = await crud.update_version_status(session, version_id, VersionStatus.DEPRECATED)
    except InvalidTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    if body.successor_version_id:
        v = await crud.update_version(session, version_id, successor_version_id=body.successor_version_id)
    try:
        from .webhooks import dispatch_webhook_event
        await dispatch_webhook_event("version.deprecated", {"id": version_id, "model_id": v.model_id})
    except Exception:
        logger.exception("Webhook dispatch failed for version.deprecated")
    return _version_to_dict(v)


@router.post("/versions/{version_id}/retire")
async def retire_version(version_id: str, session: SessionDep):
    try:
        v = await crud.update_version_status(session, version_id, VersionStatus.RETIRED)
    except InvalidTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return _version_to_dict(v)


PROMOTE_FLOW = {
    VersionStatus.DRAFT: VersionStatus.TESTING,
    VersionStatus.TESTING: VersionStatus.PUBLISHED,
}


@router.post("/versions/{version_id}/promote")
async def promote_version(version_id: str, session: SessionDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    if v.status == VersionStatus.PUBLISHED:
        return _version_to_dict(v)
    if v.status not in PROMOTE_FLOW:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot promote from {v.status.value}",
        )
    promoted_steps = []
    current_status = v.status
    while current_status in PROMOTE_FLOW:
        next_status = PROMOTE_FLOW[current_status]
        try:
            v = await crud.update_version_status(session, version_id, next_status, approval_level="l1")
        except InvalidTransition as e:
            logger.warning("Promote stopped at %s: %s", current_status.value, e)
            break
        promoted_steps.append(next_status.value)
        current_status = next_status
        if next_status == VersionStatus.PUBLISHED:
            try:
                from .webhooks import dispatch_webhook_event
                await dispatch_webhook_event(
                    "version.published",
                    {"id": version_id, "model_id": v.model_id},
                )
            except Exception:
                logger.exception("Webhook dispatch failed for version.published")
            break
    result = _version_to_dict(v)
    result["promoted_steps"] = promoted_steps
    return result


class UrlDownloadRequest(BaseModel):
    url: str
    version: str = ""
    format: ModelFormat = ModelFormat.MLX
    quantization: Quantization = Quantization.Q4
    expected_hash: str = ""
    release_notes: str = ""


def _validate_download_url(url_str: str) -> None:
    from ..ssrf import validate_external_url
    validate_external_url(url_str)


@router.post("/models/{model_id}/versions/download-url", status_code=202)
async def download_version_from_url(
    model_id: str,
    body: UrlDownloadRequest,
    session: SessionDep,
    store: StoreDep,
):
    _validate_download_url(body.url)
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    version = body.version or f"url-{uuid.uuid4().hex[:8]}"

    async def _do_download():
        downloader = ModelDownloader(storage_dir=str(store.model_version_dir(model_id, version)))
        result = await downloader.download(
            url=body.url,
            model_id=f"{model_id}_{version}",
            expected_hash=body.expected_hash,
        )
        if result.get("status") == "completed":
            from ..deps import get_session_factory
            sf = get_session_factory()
            async with sf() as s:
                await crud.create_version(
                    s,
                    model_id=model_id,
                    version=version,
                    format=body.format,
                    quantization=body.quantization,
                    file_path=result.get("path", ""),
                    file_hash=result.get("hash", body.expected_hash),
                    file_size=result.get("size_bytes", 0),
                    release_notes=body.release_notes,
                )
            logger.info("URL download completed: model=%s version=%s", model_id, version)
        else:
            logger.error(
                "URL download failed: model=%s version=%s error=%s",
                model_id, version, result.get("error", ""),
            )

    dl_key = f"{model_id}_{version}"
    _dl_task = asyncio.create_task(_do_download(), name=f"url-dl-{dl_key}")
    _running_url_downloads[dl_key] = _dl_task
    _dl_task.add_done_callback(lambda _: _running_url_downloads.pop(dl_key, None))
    return {"status": "download_started", "model_id": model_id, "version": version}


@router.get("/models/{model_id}/export")
async def export_model_tar(
    model_id: str,
    session: SessionDep,
    store: StoreDep,
):
    import io
    import tarfile

    from fastapi.responses import StreamingResponse

    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    versions, _ = await crud.list_versions(session, model_id, page=1, page_size=100)
    if not versions:
        raise HTTPException(status_code=404, detail="No versions found for model")

    model_dir = store.models_dir / model_id
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="Model files not found on disk")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        metadata = {
            "model": {
                "id": m.id, "name": m.name, "description": m.description,
                "model_type": m.model_type.value, "architecture": m.architecture,
                "params_size": m.params_size, "hf_repo": m.hf_repo,
            },
            "versions": [
                {
                    "id": v.id, "version": v.version, "format": v.format.value,
                    "quantization": v.quantization.value, "file_hash": v.file_hash,
                    "file_size": v.file_size,
                }
                for v in versions
            ],
        }
        meta_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode()
        info = tarfile.TarInfo(name="metadata.json")
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))

        for f in model_dir.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(model_dir)
                tar.add(str(f), arcname=str(arcname))

    buf.seek(0)
    logger.info("Exported model tar: model=%s versions=%d size=%d", model_id, len(versions), buf.getbuffer().nbytes)
    return StreamingResponse(
        buf,
        media_type="application/gzip",
        headers={"Content-Disposition": f"attachment; filename={m.name}.tar.gz"},
    )


@router.post("/models/import-tar", status_code=201)
async def import_model_tar(
    file: UploadFile,
    session: SessionDep,
    store: StoreDep,
):
    import io
    import tarfile

    content = await file.read()
    buf = io.BytesIO(content)

    try:
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            meta_member = None
            for member in tar.getmembers():
                if member.name == "metadata.json":
                    meta_member = member
                    break
            if not meta_member:
                raise HTTPException(status_code=400, detail="Invalid tar: missing metadata.json")

            meta_f = tar.extractfile(meta_member)
            metadata = json.loads(meta_f.read())

            model_data = metadata.get("model", {})
            name = model_data.get("name", "")
            if not name:
                raise HTTPException(status_code=400, detail="metadata.json missing model name")

            existing = await crud.get_model_by_name(session, name)
            if existing:
                raise HTTPException(status_code=409, detail=f"Model already exists: {name}")

            from ...db.models import ModelType
            try:
                mt = ModelType(model_data.get("model_type", "llm"))
            except ValueError:
                mt = ModelType.LLM

            m = await crud.create_model(
                session, name=name,
                description=model_data.get("description", ""),
                model_type=mt,
                architecture=model_data.get("architecture", ""),
                params_size=model_data.get("params_size", ""),
                hf_repo=model_data.get("hf_repo", ""),
            )

            model_dir = store.models_dir / m.id
            model_dir.mkdir(parents=True, exist_ok=True)
            model_dir_resolved = model_dir.resolve()

            for member in tar.getmembers():
                if member.name == "metadata.json":
                    continue
                # F-02 TarSlip: reject any member whose resolved path escapes the
                # model dir (absolute paths, .. traversal, symlinks). filter='data'
                # is the 3.12+ hardening; we also resolve-check as defense-in-depth.
                if member.issym() or member.islnk():
                    logger.warning("Skipping unsafe tar link member: %s", member.name)
                    continue
                target = (model_dir / member.name)
                try:
                    resolved = target.resolve(strict=False)
                except (OSError, ValueError):
                    logger.warning("Skipping unresolvable tar member: %s", member.name)
                    continue
                if not str(resolved).startswith(str(model_dir_resolved) + os.sep) and resolved != model_dir_resolved:
                    logger.warning("Skipping tar member escaping model dir: %s", member.name)
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(f.read())

            for v_data in metadata.get("versions", []):
                _version_dir = model_dir / v_data.get("version", "unknown")
                await crud.create_version(
                    session,
                    model_id=m.id,
                    version=v_data.get("version", ""),
                    file_hash=v_data.get("file_hash", ""),
                    file_size=v_data.get("file_size", 0),
                    release_notes="imported from tar",
                )

            logger.info("Imported model from tar: name=%s id=%s", name, m.id)
            return {"id": m.id, "name": m.name, "versions_imported": len(metadata.get("versions", []))}
    except tarfile.TarError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tar file: {e}")
