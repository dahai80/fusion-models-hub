import json
import logging
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ...db.models import ScanStatus
from ..deps import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["security"])

# FR-025: Security scanning for malicious code, unsafe deps, sensitive info
# Called by: app.py router registration, tests/test_api.py
# E-S7: the prior trigger_scan was a stub — it hard-coded
# malicious_code=False and set source_verified=True purely from `if model.hf_repo`,
# a free-text field any caller can populate. A non-existent scan that reports
# "clean" is worse than no scan (false assurance). Now the scan honestly reports
# its limitation: it only performs a shallow source-provenance check, and marks
# deep analysis (malicious code / unsafe deps / sensitive info) as
# "not_scanned" rather than fabricating False. source_verified requires a
# well-formed HF org/name repo id, not mere presence of the field.

_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,96}/[A-Za-z0-9][A-Za-z0-9._-]{0,96}$")


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


def _verify_hf_repo(hf_repo: str) -> bool:
    # E-S7: hf_repo is arbitrary user text; a real HF repo id is org/name.
    # Validate the shape before treating the source as provenance-verified.
    if not hf_repo:
        return False
    return bool(_HF_REPO_RE.match(hf_repo.strip()))


@router.post("/security/scan")
async def trigger_scan(body: ScanRequest, session: SessionDep):
    model = await crud.get_model(session, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    scan = await crud.create_security_scan(
        session, model_id=body.model_id,
        version_id=body.version_id, scan_type=body.scan_type,
    )
    source_verified = _verify_hf_repo(model.hf_repo or "")
    # E-S7: do NOT fabricate a clean malicious_code/deps/secrets verdict. The
    # Hub has no static-analysis engine; report what was actually inspected and
    # flag the rest as "not_scanned" so operators know deep review is pending.
    findings = {
        "source_verified": source_verified,
        "source_repo": model.hf_repo or "",
        "malicious_code": "not_scanned",
        "unsafe_dependencies": "not_scanned",
        "sensitive_info": "not_scanned",
        "note": "Shallow provenance check only; deep static analysis not implemented",
    }
    # An unverifiable source raises the floor risk; deep-analysis gaps are a
    # separate "manual_review" flag, not a risk bump on their own.
    risk_level = "low" if source_verified else "medium"
    scan = await crud.update_security_scan(
        session, scan.id,
        status=ScanStatus.COMPLETED,
        findings=json.dumps(findings),
        risk_level=risk_level,
    )
    logger.info("Security scan completed (provenance only): id=%s risk=%s verified=%s",
                scan.id, risk_level, source_verified)
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
