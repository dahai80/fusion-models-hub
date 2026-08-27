import hashlib
import hmac
import json
import logging
import os
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["watermark"])

# NFR-003: Model watermark & tracing
# Called by: app.py include_router, tests/test_api.py
# Schema: Watermark ORM in db/models.py
# E-S6: the prior default signing secret was a source-public constant, so a
# watermark could be forged by anyone with the source. Now refuse to sign unless
# a non-default FMH_WATERMARK_SECRET is configured. Verification uses a
# constant-time compare (hmac.compare_digest) so signature-mismatch does not
# leak timing.
_DEFAULT_SECRET = "fusion-model-hub-default-secret"  # noqa: S105 - sentinel, not a real secret


def _resolve_wm_secret() -> str:
    secret = os.environ.get("FMH_WATERMARK_SECRET", "")
    if not secret or secret == _DEFAULT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Watermark disabled: set a non-default FMH_WATERMARK_SECRET env "
            "(high entropy) before embedding/verifying watermarks",
        )
    return secret


class WatermarkEmbedRequest(BaseModel):
    model_id: str
    version_id: str = ""
    watermark_type: str = "metadata"
    payload: dict = {}


class WatermarkVerifyRequest(BaseModel):
    model_id: str
    version_id: str = ""
    signature: str = ""


def _wm_to_dict(w) -> dict:
    return {
        "id": w.id,
        "model_id": w.model_id,
        "version_id": w.version_id,
        "watermark_type": w.watermark_type,
        "payload": json.loads(w.payload) if w.payload else {},
        "signature": w.signature,
        "created_at": w.created_at.isoformat() if w.created_at else None,
    }


def _sign_payload(payload: dict, model_id: str, version_id: str, secret: str) -> str:
    raw = f"{secret}:{model_id}:{version_id}:{json.dumps(payload, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@router.post("/watermark/embed")
async def embed_watermark(body: WatermarkEmbedRequest, session: SessionDep, request: Request):
    model = await crud.get_model(session, body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    secret = _resolve_wm_secret()
    payload = body.payload or {}
    payload["embedded_at"] = datetime.now(UTC).isoformat()
    # E-S6: owner previously came from model.owner — a forgeable free-text field
    # any caller could set on model create. Use the authenticated tenant context
    # instead so the watermark's provenance cannot be self-asserted.
    payload["owner"] = getattr(request.state, "tenant_id", "") or model.tenant_id or ""
    signature = _sign_payload(payload, body.model_id, body.version_id, secret)
    wm = await crud.create_watermark(
        session,
        model_id=body.model_id,
        version_id=body.version_id,
        watermark_type=body.watermark_type,
        payload=json.dumps(payload),
        signature=signature,
    )
    logger.info("Watermark embedded: id=%s model=%s", wm.id, body.model_id)
    return _wm_to_dict(wm)


@router.post("/watermark/verify")
async def verify_watermark(body: WatermarkVerifyRequest, session: SessionDep):
    wms = await crud.list_watermarks(session, model_id=body.model_id, version_id=body.version_id)
    if not wms:
        return {"verified": False, "reason": "No watermark found"}
    wm = wms[0]
    secret = _resolve_wm_secret()
    payload = json.loads(wm.payload) if wm.payload else {}
    expected_sig = _sign_payload(payload, body.model_id, body.version_id, secret)
    # E-S6: constant-time compare so a signature mismatch does not leak how many
    # leading bytes matched (timing oracle for forgery).
    verified = hmac.compare_digest(wm.signature or "", expected_sig)
    return {
        "verified": verified,
        "watermark": _wm_to_dict(wm) if verified else None,
        "reason": "" if verified else "Signature mismatch",
    }


@router.get("/watermark/list")
async def list_watermarks(session: SessionDep, model_id: str = "", version_id: str = ""):
    wms = await crud.list_watermarks(session, model_id=model_id, version_id=version_id)
    return {"items": [_wm_to_dict(w) for w in wms]}
