import logging

from fastapi import APIRouter, HTTPException, Request
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
            file_path = store.get_file(oid)
            if file_path and file_path.exists():
                entry["actions"] = {
                    "download": {"href": f"{base_url}/api/v1/gitlfs/objects/{oid}"},
                }
            else:
                entry["error"] = {"code": 404, "message": "Object not found"}
        result_objects.append(entry)
    return {"objects": result_objects}


@router.post("/gitlfs/locks")
async def create_lock(body: LockRequest, session: SessionDep):
    model = await crud.get_model(session, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    existing = await crud.list_gitlfs_locks(session, model_id=body.model_id, path=body.path)
    if existing:
        raise HTTPException(status_code=409, detail="Path already locked")
    lock = await crud.create_gitlfs_lock(
        session, model_id=body.model_id, path=body.path, owner=body.owner,
    )
    return {
        "lock": {
            "id": lock.id, "path": lock.path, "owner": lock.owner,
            "locked_at": lock.created_at.isoformat() if lock.created_at else None,
        },
    }


@router.get("/gitlfs/locks")
async def list_locks(session: SessionDep, model_id: str = "", path: str = ""):
    locks = await crud.list_gitlfs_locks(session, model_id=model_id, path=path)
    return {
        "locks": [
            {
                "id": l.id, "path": l.path, "owner": l.owner,
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
