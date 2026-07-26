import logging
import uuid
from typing import Any

from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ...db import crud
from ...db.crud import InvalidTransition
from ...db.models import ModelFormat, Quantization, VersionStatus
from ...repo.downloader import ModelDownloader
from ..deps import SessionDep, StoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["versions"])


class VersionCreate(BaseModel):
    version: str
    format: ModelFormat = ModelFormat.MLX
    quantization: Quantization = Quantization.Q4
    release_notes: str = ""


class StatusChange(BaseModel):
    target_status: VersionStatus
    remark: str = ""


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
        "created_at": v.created_at.isoformat() if v.created_at else None,
    }


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
        data = await file.read()
        path, hash_val, size = await store.write_file(target_dir, file.filename or "model.bin", data)
        file_path = str(path)
        file_hash = hash_val
        file_size = size
        logger.info("Uploaded file for model=%s version=%s size=%d", model_id, version, size)

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

    upload_id = f"{model_id}_{version}"
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
    status: str = "",
    page: int = 1,
    page_size: int = 20,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    versions, total = await crud.list_versions(
        session, model_id, status=status, page=page, page_size=page_size,
    )
    return {
        "items": [_version_to_dict(v) for v in versions],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/versions/{version_id}")
async def get_version(version_id: str, session: SessionDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return _version_to_dict(v)


@router.put("/versions/{version_id}/status")
async def update_version_status(version_id: str, body: StatusChange, session: SessionDep):
    try:
        v = await crud.update_version_status(session, version_id, body.target_status)
    except InvalidTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    return _version_to_dict(v)


@router.get("/versions/{version_id}/download")
async def download_version(version_id: str, session: SessionDep, store: StoreDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
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


@router.post("/versions/{version_id}/rollback")
async def rollback_version(version_id: str, session: SessionDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    try:
        v = await crud.update_version_status(session, version_id, VersionStatus.PUBLISHED)
    except InvalidTransition as e:
        raise HTTPException(status_code=409, detail=str(e))
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


class UrlDownloadRequest(BaseModel):
    url: str
    version: str = ""
    format: ModelFormat = ModelFormat.MLX
    quantization: Quantization = Quantization.Q4
    expected_hash: str = ""
    release_notes: str = ""


def _validate_download_url(url_str: str) -> None:
    from urllib.parse import urlparse
    parsed = urlparse(url_str)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL must use http or https scheme")
    hostname = parsed.hostname or ""
    blocked = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"}
    if hostname.lower() in blocked:
        raise HTTPException(status_code=400, detail="URL cannot point to internal network")
    if hostname.startswith("10.") or hostname.startswith("192.168."):
        raise HTTPException(status_code=400, detail="URL cannot point to internal network")
    octets = hostname.split(".")
    if len(octets) == 4 and octets[0] == "172" and octets[1].isdigit() and 16 <= int(octets[1]) <= 31:
        raise HTTPException(status_code=400, detail="URL cannot point to internal network")


@router.post("/models/{model_id}/versions/download-url", status_code=202)
async def download_version_from_url(
    model_id: str,
    body: UrlDownloadRequest,
    session: SessionDep,
    store: StoreDep,
):
    import asyncio
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
                    file_hash=body.expected_hash,
                    file_size=result.get("size_bytes", 0),
                    release_notes=body.release_notes,
                )
            logger.info("URL download completed: model=%s version=%s", model_id, version)
        else:
            logger.error("URL download failed: model=%s version=%s error=%s", model_id, version, result.get("error", ""))

    asyncio.create_task(_do_download(), name=f"url-dl-{model_id}-{version}")
    return {"status": "download_started", "model_id": model_id, "version": version}
