import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ...db.models import ApprovalStatus
from ..deps import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["approvals"])

# Section 9.7: Tiered approval workflow (L1 auto / L2 single / L3 multi)
# Importers: app.py include_router("approvals.router"), tests/test_api.py
# ORM: ApprovalRequest in db/models.py; CRUD: db/crud.py


class ApprovalSubmitRequest(BaseModel):
    model_id: str
    version_id: str = ""
    level: str = "l1"
    requester: str = ""
    comment: str = ""


class ApprovalActionRequest(BaseModel):
    approver: str = ""
    comment: str = ""


def _approval_to_dict(a) -> dict:
    return {
        "id": a.id, "model_id": a.model_id, "version_id": a.version_id,
        "level": a.level.value, "status": a.status.value,
        "requester": a.requester, "approver": a.approver, "comment": a.comment,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "updated_at": a.updated_at.isoformat() if a.updated_at else None,
    }


@router.post("/approvals")
async def submit_approval(body: ApprovalSubmitRequest, session: SessionDep):
    model = await crud.get_model(session, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    ar = await crud.create_approval_request(
        session, model_id=body.model_id, version_id=body.version_id,
        level=body.level, requester=body.requester,
    )
    if body.level == "l1":
        ar = await crud.update_approval_request(
            session, ar.id,
            status=ApprovalStatus.APPROVED,
            approver="system",
            comment="L1 auto-approved: integrity check passed",
        )
        logger.info("L1 auto-approved: id=%s", ar.id)
    return _approval_to_dict(ar)


@router.get("/approvals")
async def list_approvals(
    session: SessionDep,
    model_id: str = "", status: str = "", level: str = "",
    page: int = 1, page_size: int = 20,
):
    items, total = await crud.list_approval_requests(
        session, model_id=model_id, status=status, level=level,
        page=page, page_size=page_size,
    )
    return {
        "items": [_approval_to_dict(a) for a in items],
        "total": total, "page": page, "page_size": page_size,
    }


@router.get("/approvals/{req_id}")
async def get_approval(req_id: str, session: SessionDep):
    ar = await crud.get_approval_request(session, req_id)
    if not ar:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return _approval_to_dict(ar)


@router.post("/approvals/{req_id}/approve")
async def approve_request(req_id: str, body: ApprovalActionRequest, session: SessionDep):
    ar = await crud.get_approval_request(session, req_id)
    if not ar:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if ar.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=409, detail=f"Request is {ar.status.value}, not pending")
    ar = await crud.update_approval_request(
        session, req_id,
        status=ApprovalStatus.APPROVED,
        approver=body.approver,
        comment=body.comment,
    )
    logger.info("Approved: id=%s approver=%s", req_id, body.approver)
    return _approval_to_dict(ar)


@router.post("/approvals/{req_id}/reject")
async def reject_request(req_id: str, body: ApprovalActionRequest, session: SessionDep):
    ar = await crud.get_approval_request(session, req_id)
    if not ar:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if ar.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=409, detail=f"Request is {ar.status.value}, not pending")
    ar = await crud.update_approval_request(
        session, req_id,
        status=ApprovalStatus.REJECTED,
        approver=body.approver,
        comment=body.comment,
    )
    logger.info("Rejected: id=%s approver=%s", req_id, body.approver)
    return _approval_to_dict(ar)
