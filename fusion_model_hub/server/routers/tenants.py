import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants", tags=["tenants"])


def _require_admin(request: Request) -> None:
    # E-S8: tenant create/update/delete is an admin-only privilege. Without this
    # any developer-role key could create/rename tenants and even delete one
    # (the prior code had no RBAC at all on these endpoints).
    from ..auth import _is_auth_enabled
    if not _is_auth_enabled():
        return
    role = getattr(request.state, "user_role", "")
    if role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can manage tenants")


class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    display_name: str = Field("", max_length=128)


class TenantUpdate(BaseModel):
    display_name: str | None = None


class TenantOut(BaseModel):
    id: str
    name: str
    display_name: str
    is_active: bool
    model_config = {"from_attributes": True}


@router.post("", status_code=201, response_model=TenantOut)
async def create_tenant(body: TenantCreate, session: SessionDep, request: Request):
    _require_admin(request)
    existing = await crud.get_tenant_by_name(session, body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Tenant already exists: {body.name}")
    t = await crud.create_tenant(session, name=body.name, display_name=body.display_name)
    return t


@router.get("")
async def list_tenants(session: SessionDep, request: Request):
    _require_admin(request)
    items = await crud.list_tenants(session)
    logger.info("Listed tenants: %d", len(items))
    return {"tenants": items, "total": len(items)}


@router.get("/{tenant_id}", response_model=TenantOut)
async def get_tenant(tenant_id: str, session: SessionDep, request: Request):
    _require_admin(request)
    t = await crud.get_tenant(session, tenant_id)
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return t


@router.patch("/{tenant_id}", response_model=TenantOut)
async def update_tenant(tenant_id: str, body: TenantUpdate, session: SessionDep, request: Request):
    _require_admin(request)
    t = await crud.get_tenant(session, tenant_id)
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    if body.display_name is not None:
        t.display_name = body.display_name
    await session.commit()
    await session.refresh(t)
    logger.info("Updated tenant: id=%s", tenant_id)
    return t


@router.delete("/{tenant_id}")
async def delete_tenant(tenant_id: str, session: SessionDep, request: Request):
    _require_admin(request)
    # E-S8: refuse to delete a tenant that still owns models or API keys — a
    # blind cascade orphans those rows (model.tenant_id pointing at a gone
    # tenant). The caller must reassign or delete dependents first.
    owned = await crud.count_models_for_tenant(session, tenant_id=tenant_id)
    if owned:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete tenant: {owned} model(s) still assigned. "
                   "Reassign or delete them first.",
        )
    keys = await crud.count_api_keys_for_tenant(session, tenant_id=tenant_id)
    if keys:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete tenant: {keys} API key(s) still active. "
                   "Deactivate them first.",
        )
    ok = await crud.delete_tenant(session, tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"detail": "deleted"}


# -- Tenant Roles --


class RoleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    permissions: str = Field("read", max_length=512)


class RoleUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=64)
    permissions: str | None = Field(None, max_length=512)
    is_active: bool | None = None


class RoleOut(BaseModel):
    id: str
    tenant_id: str
    name: str
    permissions: str
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None
    model_config = {"from_attributes": True}


def _role_to_dict(r) -> dict:
    return {
        "id": r.id,
        "tenant_id": r.tenant_id,
        "name": r.name,
        "permissions": r.permissions,
        "is_active": r.is_active,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


@router.get("/{tenant_id}/roles")
async def list_roles(tenant_id: str, session: SessionDep):
    roles = await crud.list_roles(session, tenant_id)
    return {"items": [_role_to_dict(r) for r in roles]}


@router.post("/{tenant_id}/roles", status_code=201)
async def create_role(tenant_id: str, body: RoleCreate, session: SessionDep):
    t = await crud.get_tenant(session, tenant_id)
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    r = await crud.create_role(
        session, tenant_id=tenant_id, name=body.name, permissions=body.permissions,
    )
    return _role_to_dict(r)


@router.put("/{tenant_id}/roles/{role_id}")
async def update_role(tenant_id: str, role_id: str, body: RoleUpdate, session: SessionDep):
    r = await crud.get_role(session, role_id)
    if not r or r.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Role not found")
    r = await crud.update_role(
        session, role_id,
        name=body.name, permissions=body.permissions, is_active=body.is_active,
    )
    return _role_to_dict(r)


@router.delete("/{tenant_id}/roles/{role_id}")
async def delete_role(tenant_id: str, role_id: str, session: SessionDep):
    r = await crud.get_role(session, role_id)
    if not r or r.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Role not found")
    ok = await crud.delete_role(session, role_id)
    return {"detail": "deleted", "id": role_id} if ok else {"detail": "not found"}
