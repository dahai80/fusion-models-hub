import logging
import os

import anyio
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...db import crud
from ..deps import SessionDep, SettingsDep, StoreDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])


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
async def push_to_remote(body: SyncPushRequest, session: SessionDep, settings: SettingsDep):
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
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
            pushed.append({"version_id": vid, "status": "pushed", "remote_status": resp.status_code})
            logger.info("Pushed version %s to %s", vid, body.target_url)
        except Exception as e:
            pushed.append({"version_id": vid, "status": "failed", "error": str(e)})
            logger.warning("Push failed for version %s: %s", vid, e)
    return {"pushed": pushed, "count": len(pushed)}


@router.post("/pull")
async def pull_from_remote(body: SyncPullRequest, session: SessionDep, settings: SettingsDep):
    _validate_sync_url(body.source_url)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{body.source_url}/api/v1/models/{body.model_id}")
            resp.raise_for_status()
            remote_model = resp.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch remote model: {e}")

    existing = await crud.get_model_by_name(session, remote_model.get("name", ""))
    if existing:
        return {"status": "already_exists", "model_id": existing.id}

    from ...db.models import ModelType

    try:
        mt = ModelType(remote_model.get("model_type", "llm"))
    except ValueError:
        mt = ModelType.LLM
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
                file_size=rv.get("file_size", 0),
            )
        except crud.VersionConflictError:
            # P1-D: duplicate version on a re-sync — skip rather than 500.
            logger.warning("Sync: version %s/%s already exists, skipping", new_m.id, rv.get("version", ""))
            continue
        if v:
            pulled_versions.append(v.id)
    logger.info("Pulled model %s with %d versions from %s", new_m.id, len(pulled_versions), body.source_url)
    return {
        "status": "pulled",
        "model_id": new_m.id,
        "versions_pulled": len(pulled_versions),
        "version_ids": pulled_versions,
    }
