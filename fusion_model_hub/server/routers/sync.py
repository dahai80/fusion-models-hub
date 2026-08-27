import contextlib
import hashlib
import logging
import os

import anyio
import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ...db import crud
from ..deps import SessionDep, SettingsDep, StoreDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])

# #2: streaming chunk size for push/pull file bytes — 1MB keeps memory flat
# while hashing + writing without blocking the loop on a single giant read.
_SYNC_STREAM_CHUNK = 1024 * 1024


def _disk_hash_and_size(file_path: str) -> tuple[str, int]:
    # E-E8: delegate to the shared utils helper. Prior 8KB chunk was the odd
    # one out among 64KB callers; unified helper uses 64KB everywhere.
    from ...utils.hashing import compute_sha256_and_size

    return compute_sha256_and_size(file_path)


@router.get("/versions/{version_id}/manifest")
async def get_version_manifest(version_id: str, session: SessionDep, store: StoreDep):
    v = await crud.get_version(session, version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    manifest = {
        "version_id": v.id,
        "model_id": v.model_id,
        "version": v.version,
        "format": v.format.value,
        "quantization": v.quantization.value,
        "file_hash": v.file_hash,
        "file_size": v.file_size,
        "status": v.status.value,
    }
    if v.file_path and os.path.exists(v.file_path):
        disk_hash, disk_size = await anyio.to_thread.run_sync(_disk_hash_and_size, v.file_path)
        manifest["disk_hash"] = disk_hash
        manifest["disk_size"] = disk_size
    elif hasattr(store, "get_file"):
        fpath = store.get_file(v.file_path)
        if fpath and os.path.exists(str(fpath)):
            stat = os.stat(str(fpath))
            manifest["disk_size"] = stat.st_size
    logger.info("Manifest generated: version=%s hash=%s", version_id, manifest.get("file_hash", ""))
    return manifest


class SyncPushRequest(BaseModel):
    target_url: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    version_ids: list[str] = Field(default_factory=list)


class SyncPullRequest(BaseModel):
    source_url: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    version_ids: list[str] = Field(default_factory=list)


def _validate_sync_url(url_str: str) -> None:
    from ..ssrf import validate_external_url

    validate_external_url(url_str)


@router.post("/push")
async def push_to_remote(
    body: SyncPushRequest,
    session: SessionDep,
    settings: SettingsDep,
    store: StoreDep,
):
    # #2: push now streams the real weight bytes, not just metadata. Per version:
    #   1. POST metadata to /system/import (idempotent create-or-skip remote model).
    #   2. If the local version has a file, stream it to the remote /sync/receive,
    #      which writes it into the remote store + sets file_path/hash/size.
    _validate_sync_url(body.target_url)
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    version_ids = body.version_ids or [v.id for v in m.versions]
    pushed = []
    for vid in version_ids:
        v = await crud.get_version(session, vid)
        if not v or v.model_id != body.model_id:
            continue
        entry: dict = {"version_id": vid, "version": v.version}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 1. metadata (existing path — backward compatible).
                resp = await client.post(
                    f"{body.target_url}/api/v1/system/import",
                    json={
                        "models": [
                            {
                                "id": m.id,
                                "name": m.name,
                                "description": m.description,
                                "model_type": m.model_type.value,
                                "architecture": m.architecture,
                                "params_size": m.params_size,
                                "hf_repo": m.hf_repo,
                            }
                        ]
                    },
                )
                entry["metadata_status"] = resp.status_code

                # 2. real file bytes — only if the local version has one.
                if v.file_path and os.path.exists(v.file_path):
                    file_path = store.get_file(v.file_path) or v.file_path
                    if file_path and os.path.exists(str(file_path)):
                        with open(file_path, "rb") as fh:
                            files = {
                                "file": (
                                    os.path.basename(str(file_path)),
                                    fh,
                                    "application/octet-stream",
                                ),
                            }
                            file_resp = await client.post(
                                f"{body.target_url}/api/v1/sync/receive",
                                data={
                                    "model_id": m.id,
                                    "model_name": m.name,
                                    "version": v.version,
                                    "format": v.format.value,
                                    "quantization": v.quantization.value,
                                },
                                files=files,
                            )
                        entry["file_status"] = file_resp.status_code
                        entry["status"] = "pushed" if file_resp.status_code in (200, 201) else "partial"
                    else:
                        entry["status"] = "metadata_only"
                        entry["reason"] = "local file missing on disk"
                else:
                    entry["status"] = "metadata_only"
                    entry["reason"] = "no file_path on version"
            logger.info("Pushed version %s to %s: %s", vid, body.target_url, entry.get("status"))
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            logger.warning("Push failed for version %s: %s", vid, e)
        pushed.append(entry)
    return {"pushed": pushed, "count": len(pushed)}


@router.post("/receive")
async def receive_sync_file(
    model_id: str = Form(...),
    model_name: str = Form(...),
    version: str = Form(...),
    format: str = Form("mlx"),
    quantization: str = Form("4bit"),
    file: UploadFile = File(...),
    session: SessionDep = None,
    store: StoreDep = None,
    settings: SettingsDep = None,
):
    # #2: remote-side receive endpoint for push. Idempotent: get-or-create the
    # model (by pushed id, fall back to name) + get-or-create the version, then
    # stream the uploaded file into the local store and set file_path/hash/size.
    if not version:
        raise HTTPException(status_code=400, detail="version is required")
    m = await crud.get_model(session, model_id)
    if not m:
        existing = await crud.get_model_by_name(session, model_name)
        if existing:
            m = existing
        else:
            from ...db.models import ModelType

            try:
                mt = ModelType("llm")
            except ValueError:
                mt = ModelType.LLM
            m = await crud.create_model(
                session,
                name=model_name,
                model_type=mt,
            )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    target_dir = store.model_version_dir(m.id, version)
    safe_name = os.path.basename(file.filename or "model.bin") or "model.bin"
    hasher = hashlib.sha256()
    written = 0
    from ...storage.local_store import LocalStore

    if isinstance(store, LocalStore):
        target_path = target_dir / safe_name
        try:
            with open(target_path, "wb") as out:
                while True:
                    block = await file.read(_SYNC_STREAM_CHUNK)
                    if not block:
                        break
                    written += len(block)
                    if written > max_bytes:
                        out.close()
                        with contextlib.suppress(OSError):
                            target_path.unlink()
                        raise HTTPException(status_code=413, detail="Sync file exceeds max_upload_size_mb")
                    out.write(block)
                    hasher.update(block)
        except BaseException:
            # Crash mid-stream: do not leave a partial weight file at the path.
            with contextlib.suppress(OSError):
                target_path.unlink()
            raise
        file_hash = hasher.hexdigest()
        file_size = written
        file_path = str(target_path)
    else:
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail="Sync file exceeds max_upload_size_mb")
        target_path, file_hash, file_size = await store.write_file(target_dir, safe_name, data)
        file_path = str(target_path)

    # #2: capture the model id as a plain string before create_version — a
    # VersionConflictError rolls the session back, which expires `m`, and a
    # later `m.id` access would lazy-refresh sync (MissingGreenlet).
    model_id_str = m.id
    try:
        v = await crud.create_version(
            session,
            model_id=model_id_str,
            version=version,
            format=format,
            quantization=quantization,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
        )
    except crud.VersionConflictError:
        # Idempotent re-push: update the existing version's file fields instead
        # of 409, so a retry overwrites the bytes rather than failing.
        existing_v = await _find_version_by_label(session, model_id_str, version)
        if not existing_v:
            raise HTTPException(
                status_code=409, detail=f"Version {version!r} exists but could not be located"
            ) from None
        await crud.update_version(
            session,
            existing_v.id,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
        )
        v = existing_v
        logger.info(
            "Sync receive overwrote existing version: model=%s version=%s size=%d", model_id_str, version, file_size
        )
    logger.info(
        "Sync receive stored file: model=%s version=%s size=%d hash=%s",
        model_id_str,
        version,
        file_size,
        file_hash[:16],
    )
    return {
        "status": "received",
        "model_id": model_id_str,
        "version_id": v.id,
        "file_hash": file_hash,
        "file_size": file_size,
    }


async def _find_version_by_label(session, model_id: str, version: str):
    # crud has no get-by-label, and reading m.versions would lazy-load the
    # relationship — which throws MissingGreenlet on a session already poisoned
    # by the VersionConflictError flush. Query the version row directly.
    from sqlalchemy import select

    from ...db.models import ModelVersion

    result = await session.execute(
        select(ModelVersion).where(ModelVersion.model_id == model_id, ModelVersion.version == version)
    )
    return result.scalar_one_or_none()


@router.post("/pull")
async def pull_from_remote(
    body: SyncPullRequest,
    session: SessionDep,
    settings: SettingsDep,
    store: StoreDep,
):
    # #2: pull now streams the real weight bytes. Per remote version:
    #   1. GET {source}/api/v1/models/{id} for the model metadata + version list.
    #   2. get-or-create the local model.
    #   3. get-or-create the local version row; if the remote version has a
    #      file_path, stream GET {source}/api/v1/versions/{remote_vid}/download
    #      into the local store + set file_path/hash/size.
    _validate_sync_url(body.source_url)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{body.source_url}/api/v1/models/{body.model_id}")
            resp.raise_for_status()
            remote_model = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch remote model: {e}")

    from ...db.models import ModelType

    try:
        mt = ModelType(remote_model.get("model_type", "llm"))
    except ValueError:
        mt = ModelType.LLM
    # get-or-create the local model (idempotent re-pull).
    new_m = await crud.get_model_by_name(session, remote_model.get("name", ""))
    if not new_m:
        new_m = await crud.create_model(
            session,
            name=remote_model["name"],
            description=remote_model.get("description", ""),
            model_type=mt,
            architecture=remote_model.get("architecture", ""),
            params_size=remote_model.get("params_size", ""),
            hf_repo=remote_model.get("hf_repo", ""),
        )

    pulled_versions = []
    remote_versions = remote_model.get("versions", [])
    for rv in remote_versions:
        if body.version_ids and rv.get("id") not in body.version_ids:
            continue
        try:
            v = await crud.create_version(
                session,
                model_id=new_m.id,
                version=rv.get("version", ""),
                format=rv.get("format", "mlx"),
                quantization=rv.get("quantization", "4bit"),
            )
        except crud.VersionConflictError:
            # P1-D: duplicate version on a re-sync — skip rather than 500.
            logger.warning("Sync: version %s/%s already exists, skipping", new_m.id, rv.get("version", ""))
            continue

        # Stream the remote file bytes into the local store.
        remote_vid = rv.get("id", "")
        remote_has_file = rv.get("file_path") or rv.get("file_size", 0) > 0
        if remote_vid and remote_has_file:
            await _stream_remote_file(
                body.source_url,
                remote_vid,
                store,
                new_m.id,
                rv.get("version", ""),
                settings,
                session,
                v.id,
            )
        pulled_versions.append(v.id)
    logger.info("Pulled model %s with %d versions from %s", new_m.id, len(pulled_versions), body.source_url)
    return {
        "status": "pulled",
        "model_id": new_m.id,
        "versions_pulled": len(pulled_versions),
        "version_ids": pulled_versions,
    }


async def _stream_remote_file(
    source_url: str,
    remote_version_id: str,
    store: StoreDep,
    model_id: str,
    version: str,
    settings: SettingsDep,
    session,
    local_version_id: str,
):
    # #2: stream GET {source}/api/v1/versions/{id}/download into the local store,
    # hashing as we go, then update the local version's file_path/hash/size.
    from ...storage.local_store import LocalStore

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            dl_resp = await client.get(f"{source_url}/api/v1/versions/{remote_version_id}/download")
            if dl_resp.status_code != 200:
                logger.warning(
                    "Sync pull: remote download %s -> %s, skipping file",
                    remote_version_id,
                    dl_resp.status_code,
                )
                return
            target_dir = store.model_version_dir(model_id, version)
            # Reuse the remote filename if the server sent one, else model.bin.
            cd = dl_resp.headers.get("content-disposition", "")
            fname = "model.bin"
            if "filename=" in cd:
                fname = cd.split("filename=")[-1].strip('"').strip()
            safe_name = os.path.basename(fname) or "model.bin"
            hasher = hashlib.sha256()
            written = 0
            if isinstance(store, LocalStore):
                target_path = target_dir / safe_name
                try:
                    with open(target_path, "wb") as out:
                        async for block in dl_resp.aiter_bytes(_SYNC_STREAM_CHUNK):
                            written += len(block)
                            if written > max_bytes:
                                out.close()
                                with contextlib.suppress(OSError):
                                    target_path.unlink()
                                raise HTTPException(status_code=413, detail="Sync pull file exceeds max_upload_size_mb")
                            out.write(block)
                            hasher.update(block)
                except BaseException:
                    with contextlib.suppress(OSError):
                        target_path.unlink()
                    raise
                file_hash = hasher.hexdigest()
                file_size = written
                file_path = str(target_path)
            else:
                data = b"".join([block async for block in dl_resp.aiter_bytes(_SYNC_STREAM_CHUNK)])
                if len(data) > max_bytes:
                    raise HTTPException(status_code=413, detail="Sync pull file exceeds max_upload_size_mb")
                target_path, file_hash, file_size = await store.write_file(target_dir, safe_name, data)
                file_path = str(target_path)
            await crud.update_version(
                session,
                local_version_id,
                file_path=file_path,
                file_hash=file_hash,
                file_size=file_size,
            )
            logger.info(
                "Sync pull streamed file: model=%s version=%s size=%d hash=%s",
                model_id,
                version,
                file_size,
                file_hash[:16],
            )
    except httpx.HTTPError as e:
        logger.warning("Sync pull: failed to stream remote file %s: %s", remote_version_id, e)
