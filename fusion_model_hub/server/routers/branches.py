import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...db import crud
from ...db.crud import VersionConflictError
from ...db.models import BranchStatus
from ..deps import SessionDep, get_session_factory

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
    # R-P2/#5: a merge must promote the branch head into the model's mainline
    # version history, not merely flip the branch status. Before, merge left no
    # new version row — the head's work was invisible to the version list, so
    # inference never served it and a "merged" branch pointed at nothing. Now
    # the head version (required) is copied into a new mainline version with a
    # merge-suffixed label, and the branch flips to MERGED. The branch stays
    # head-less-safe: a branch never given a head_version_id is rejected with a
    # clear 400 instead of silently merging nothing.
    head_id = (b.head_version_id or "").strip()
    if not head_id:
        raise HTTPException(
            status_code=400,
            detail="Branch has no head_version_id — set a head before merging",
        )
    head = await crud.get_version(session, head_id)
    if not head:
        raise HTTPException(status_code=404, detail="Branch head version not found")
    # Snapshot the head's immutable scalar fields BEFORE any version create, so
    # a create_version rollback (on a duplicate-label conflict) cannot expire
    # the head object and trigger a lazy refresh outside an async context
    # (MissingGreenlet). Reading them now, while the session is healthy, keeps
    # the merge logic independent of the create session's transaction state.
    head_model_id = head.model_id
    head_version_label = head.version
    head_format = head.format
    head_quantization = head.quantization
    head_file_path = head.file_path
    head_file_hash = head.file_hash
    head_file_size = head.file_size
    branch_name = b.name
    merge_label = f"{head_version_label}-merge-{branch_name}"[:32]
    # R-P2/#5: do the version create in its OWN session so a duplicate-label
    # conflict (idempotent re-merge) rolls back only that throwaway session,
    # never the router's main session. Before, the conflict rollback poisoned
    # the shared session and the follow-up update_model_branch raised
    # MissingGreenlet -> 500 on every re-merge.
    promoted_id = ""
    factory = get_session_factory()
    try:
        async with factory() as vsession:
            promoted = await crud.create_version(
                vsession,
                model_id=head_model_id,
                version=merge_label,
                format=head_format,
                quantization=head_quantization,
                file_path=head_file_path,
                file_hash=head_file_hash,
                file_size=head_file_size,
                release_notes=f"Merged from branch {branch_name!r} (head {head_version_label})",
            )
            promoted_id = promoted.id if promoted else ""
    except VersionConflictError:
        # Idempotent re-merge: the merge label already exists. Re-fetch it on a
        # fresh session so a repeated merge call still returns 200 with the
        # existing version id.
        async with factory() as vsession:
            existing = await crud.get_version_by_label(vsession, head_model_id, merge_label)
            promoted_id = existing.id if existing else ""
        logger.info("Branch re-merge (version existed): id=%s label=%s", branch_id, merge_label)
    if not promoted_id:
        raise HTTPException(status_code=500, detail="Failed to promote merged version")
    b = await crud.update_model_branch(
        session, branch_id, status=BranchStatus.MERGED,
    )
    logger.info("Merged branch: id=%s name=%s promoted_version=%s", branch_id, b.name, promoted_id)
    result = _branch_to_dict(b)
    result["merged_version_id"] = promoted_id
    return result
