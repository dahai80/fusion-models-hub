import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants", tags=["tenants"])


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
async def create_tenant(body: TenantCreate, session: SessionDep):
    existing = await crud.get_tenant_by_name(session, body.name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Tenant already exists: {body.name}")
    t = await crud.create_tenant(session, name=body.name, display_name=body.display_name)
    return t


@router.get("", response_model=list[TenantOut])
async def list_tenants(session: SessionDep):
    return await crud.list_tenants(session)


@router.get("/{tenant_id}", response_model=TenantOut)
async def get_tenant(tenant_id: str, session: SessionDep):
    t = await crud.get_tenant(session, tenant_id)
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return t


@router.patch("/{tenant_id}", response_model=TenantOut)
async def update_tenant(tenant_id: str, body: TenantUpdate, session: SessionDep):
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
async def delete_tenant(tenant_id: str, session: SessionDep):
    ok = await crud.delete_tenant(session, tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return {"detail": "deleted"}
