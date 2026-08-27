import logging
import os

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep, StoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["gitlfs"])

# FR-027: Git LFS v2 batch API + lock management
# Importers: app.py include_router, tests/test_api.py
# ORM: GitLfsLock in db/models.py


class BatchRequest(BaseModel):
    operation: str
    objects: list[dict]
    transfers: list[str] = ["basic"]
    ref: dict = {}


class LockRequest(BaseModel):
    model_id: str
    path: str
    owner: str = ""


class VerifyRequest(BaseModel):
    oid: str
    size: int


@router.post("/gitlfs/objects/batch")
async def batch_api(body: BatchRequest, session: SessionDep, store: StoreDep, request: Request):
    base_url = str(request.base_url).rstrip("/")
    result_objects = []
    for obj in body.objects:
        oid = obj.get("oid", "")
        size = obj.get("size", 0)
        entry = {"oid": oid, "size": size}
        if body.operation == "upload":
            entry["actions"] = {
                "upload": {
                    "href": f"{base_url}/api/v1/gitlfs/objects/{oid}",
                    "header": {"Content-Type": "application/octet-stream"},
                },
                "verify": {"href": f"{base_url}/api/v1/gitlfs/verify"},
            }
        elif body.operation == "download":
            # P1-2: resolve the object via the LFS store, not the generic
            # get_file (which treated oid as an arbitrary path and could miss).
            file_path = store.get_lfs_object(oid)
            if file_path and file_path.exists():
                entry["actions"] = {
                    "download": {"href": f"{base_url}/api/v1/gitlfs/objects/{oid}"},
                }
            else:
                entry["error"] = {"code": 404, "message": "Object not found"}
        result_objects.append(entry)
    return {"objects": result_objects}


@router.put("/gitlfs/objects/{oid}")
async def upload_object(oid: str, request: Request, store: StoreDep):
    # P1-2: the batch API advertises this endpoint for uploads. Before this
    # route existed the href was a phantom 404 — LFS clients could not push.
    # oid must be a bare hash; reject path-traversal shapes early.
    if os.path.basename(oid) != oid or not oid:
        raise HTTPException(status_code=400, detail="Invalid oid")
    data = await request.body()
    try:
        store.put_lfs_object(oid, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    logger.info("LFS upload ok: oid=%s size=%d", oid[:16], len(data))
    return Response(status_code=200, media_type="application/json", content=f'{{"oid":"{oid}","size":{len(data)}}}')


@router.get("/gitlfs/objects/{oid}")
async def download_object(oid: str, store: StoreDep):
    if os.path.basename(oid) != oid or not oid:
        raise HTTPException(status_code=400, detail="Invalid oid")
    file_path = store.get_lfs_object(oid)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Object not found")
    data = file_path.read_bytes()
    logger.info("LFS download ok: oid=%s size=%d", oid[:16], len(data))
    return Response(content=data, media_type="application/octet-stream")


@router.post("/gitlfs/verify")
async def verify_object(body: VerifyRequest, store: StoreDep):
    if os.path.basename(body.oid) != body.oid or not body.oid:
        raise HTTPException(status_code=400, detail="Invalid oid")
    file_path = store.get_lfs_object(body.oid)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Object not found")
    actual_size = file_path.stat().st_size
    if actual_size != body.size:
        logger.warning("LFS verify size mismatch: oid=%s expected=%d actual=%d", body.oid[:16], body.size, actual_size)
        raise HTTPException(status_code=422, detail="Size mismatch")
    logger.info("LFS verify ok: oid=%s size=%d", body.oid[:16], body.size)
    return {"oid": body.oid, "size": actual_size}


@router.post("/gitlfs/locks")
async def create_lock(body: LockRequest, session: SessionDep):
    model = await crud.get_model(session, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    existing = await crud.list_gitlfs_locks(session, model_id=body.model_id, path=body.path)
    if existing:
        raise HTTPException(status_code=409, detail="Path already locked")
    lock = await crud.create_gitlfs_lock(
        session,
        model_id=body.model_id,
        path=body.path,
        owner=body.owner,
    )
    return {
        "lock": {
            "id": lock.id,
            "path": lock.path,
            "owner": lock.owner,
            "locked_at": lock.created_at.isoformat() if lock.created_at else None,
        },
    }


@router.get("/gitlfs/locks")
async def list_locks(session: SessionDep, model_id: str = "", path: str = ""):
    locks = await crud.list_gitlfs_locks(session, model_id=model_id, path=path)
    return {
        "locks": [
            {
                "id": l.id,
                "path": l.path,
                "owner": l.owner,
                "locked_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in locks
        ],
    }


@router.delete("/gitlfs/locks/{lock_id}")
async def delete_lock(lock_id: str, session: SessionDep):
    ok = await crud.delete_gitlfs_lock(session, lock_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Lock not found")
    return {"lock": {"id": lock_id}, "message": "Lock removed"}
