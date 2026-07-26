import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])


class ApiKeyCreate(BaseModel):
    name: str
    permissions: str = "read,write"


@router.post("/auth/keys", status_code=201)
async def create_key(body: ApiKeyCreate, session: SessionDep):
    ak, full_key = await crud.create_api_key(
        session, name=body.name, permissions=body.permissions,
    )
    return {
        "id": ak.id,
        "name": ak.name,
        "key": full_key,
        "key_prefix": ak.key_prefix,
        "permissions": ak.permissions,
        "is_active": ak.is_active,
        "created_at": ak.created_at.isoformat() if ak.created_at else None,
    }


@router.get("/auth/keys")
async def list_keys(session: SessionDep):
    keys = await crud.list_api_keys(session)
    return {
        "items": [
            {
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "permissions": k.permissions,
                "is_active": k.is_active,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            }
            for k in keys
        ]
    }


@router.delete("/auth/keys/{key_id}")
async def delete_key(key_id: str, session: SessionDep):
    deleted = await crud.delete_api_key(session, key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted", "id": key_id}


@router.post("/auth/keys/{key_id}/deactivate")
async def deactivate_key(key_id: str, session: SessionDep):
    ak = await crud.deactivate_api_key(session, key_id)
    if not ak:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"id": ak.id, "is_active": ak.is_active}
