import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...db import crud
from ...db.models import BranchStatus
from ..deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["branches"])


class BranchCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    base_version_id: str = ""
    description: str = ""


class BranchUpdate(BaseModel):
    head_version_id: str | None = None
    status: str | None = None
    description: str | None = None


def _branch_to_dict(b) -> dict:
    return {
        "id": b.id,
        "model_id": b.model_id,
        "name": b.name,
        "base_version_id": b.base_version_id,
        "head_version_id": b.head_version_id,
        "status": b.status.value,
        "description": b.description,
        "created_at": b.created_at.isoformat() if b.created_at else None,
        "updated_at": b.updated_at.isoformat() if b.updated_at else None,
    }


@router.post("/{model_id}/branches", status_code=201)
async def create_branch(model_id: str, body: BranchCreate, session: SessionDep):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    b = await crud.create_model_branch(
        session, model_id=model_id, name=body.name,
        base_version_id=body.base_version_id, description=body.description,
    )
    return _branch_to_dict(b)


@router.get("/{model_id}/branches")
async def list_branches(model_id: str, session: SessionDep, status: str = ""):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    branches = await crud.list_model_branches(session, model_id=model_id, status=status)
    return {"items": [_branch_to_dict(b) for b in branches]}


@router.get("/branches/{branch_id}")
async def get_branch(branch_id: str, session: SessionDep):
    b = await crud.get_model_branch(session, branch_id)
    if not b:
        raise HTTPException(status_code=404, detail="Branch not found")
    return _branch_to_dict(b)


@router.patch("/branches/{branch_id}")
async def update_branch(branch_id: str, body: BranchUpdate, session: SessionDep):
    fields = {}
    if body.head_version_id is not None:
        fields["head_version_id"] = body.head_version_id
    if body.status is not None:
        try:
            BranchStatus(body.status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")
        fields["status"] = BranchStatus(body.status)
    if body.description is not None:
        fields["description"] = body.description
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    b = await crud.update_model_branch(session, branch_id, **fields)
    if not b:
        raise HTTPException(status_code=404, detail="Branch not found")
    return _branch_to_dict(b)


@router.delete("/branches/{branch_id}")
async def delete_branch(branch_id: str, session: SessionDep):
    ok = await crud.delete_model_branch(session, branch_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Branch not found")
    return {"detail": "deleted"}


@router.post("/branches/{branch_id}/merge")
async def merge_branch(branch_id: str, session: SessionDep):
    b = await crud.get_model_branch(session, branch_id)
    if not b:
        raise HTTPException(status_code=404, detail="Branch not found")
    if b.status != BranchStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Only active branches can be merged")
    b = await crud.update_model_branch(
        session, branch_id, status=BranchStatus.MERGED,
    )
    logger.info("Merged branch: id=%s name=%s", branch_id, b.name)
    return _branch_to_dict(b)
