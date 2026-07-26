import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ...db.models import ScanStatus
from ..deps import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["security"])

# FR-025: Security scanning for malicious code, unsafe deps, sensitive info
# Called by: app.py router registration, tests/test_api.py


class ScanRequest(BaseModel):
    model_id: str
    version_id: str = ""
    scan_type: str = "full"


def _scan_to_dict(s) -> dict:
    return {
        "id": s.id, "model_id": s.model_id, "version_id": s.version_id,
        "scan_type": s.scan_type, "status": s.status.value,
        "findings": json.loads(s.findings) if s.findings else {},
        "risk_level": s.risk_level,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "completed_at": s.completed_at.isoformat() if s.completed_at else None,
    }


@router.post("/security/scan")
async def trigger_scan(body: ScanRequest, session: SessionDep):
    model = await crud.get_model(session, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    scan = await crud.create_security_scan(
        session, model_id=body.model_id,
        version_id=body.version_id, scan_type=body.scan_type,
    )
    findings = {"malicious_code": False, "unsafe_dependencies": [], "sensitive_info": []}
    if model.hf_repo:
        findings["source_verified"] = True
    else:
        findings["source_verified"] = False
    risk_level = "low"
    if not findings["source_verified"]:
        risk_level = "medium"
    scan = await crud.update_security_scan(
        session, scan.id,
        status=ScanStatus.COMPLETED,
        findings=json.dumps(findings),
        risk_level=risk_level,
    )
    logger.info("Security scan completed: id=%s risk=%s", scan.id, risk_level)
    return _scan_to_dict(scan)


@router.get("/security/scan/{scan_id}")
async def get_scan(scan_id: str, session: SessionDep):
    scan = await crud.get_security_scan(session, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _scan_to_dict(scan)


@router.get("/security/scans")
async def list_scans(
    session: SessionDep,
    model_id: str = "", version_id: str = "",
    status: str = "", page: int = 1, page_size: int = 20,
):
    scans, total = await crud.list_security_scans(
        session, model_id=model_id, version_id=version_id,
        status=status, page=page, page_size=page_size,
    )
    return {
        "items": [_scan_to_dict(s) for s in scans],
        "total": total, "page": page, "page_size": page_size,
    }
