import hashlib
import hmac
import json
import logging
import os
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep, StoreDep

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

# #1: the sidecar filename written into the version dir so the watermark
# travels with the model files and verifies without the Hub DB.
_SIDECAR_NAME = "watermark.json"


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


def _sidecar_payload(wm_dict: dict) -> bytes:
    # #1: the sidecar is the watermark dict (id/model_id/version_id/type/
    # payload/signature/created_at) as canonical JSON. The signature inside
    # already binds (model_id, version_id, payload, secret), so a tampered
    # sidecar fails the HMAC re-verify without the DB row.
    return json.dumps(wm_dict, sort_keys=True).encode()


async def _resolve_version_label(session, model_id: str, version_id: str) -> str:
    # #1: the storage path is models/{model_id}/{version_str}/, but the request
    # carries version_id (the row id). Resolve to the version string; fall back
    # to version_id so a watermark can still be embedded when no version row is
    # linked (watermark_type=metadata on the model itself).
    if not version_id:
        return "default"
    v = await crud.get_version(session, version_id)
    if v and v.version:
        return v.version
    return version_id


@router.post("/watermark/embed")
async def embed_watermark(
    body: WatermarkEmbedRequest,
    session: SessionDep,
    store: StoreDep,
    request: Request,
):
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
    wm_dict = _wm_to_dict(wm)
    # #1: also write a signed sidecar into the version dir so the watermark
    # travels with the model files. A backend that cannot (MinioStore) raises
    # NotImplementedError — surface as 501, but the DB row is already written
    # so the watermark is still verifiable via the DB path.
    sidecar_written = False
    try:
        version_label = await _resolve_version_label(session, body.model_id, body.version_id)
        store.write_sidecar(body.model_id, version_label, _SIDECAR_NAME, _sidecar_payload(wm_dict))
        sidecar_written = True
    except NotImplementedError:
        logger.warning("Sidecar watermark not supported by storage backend: model=%s", body.model_id)
    except Exception:
        logger.exception("Failed to write watermark sidecar: model=%s version=%s", body.model_id, body.version_id)
    logger.info(
        "Watermark embedded: id=%s model=%s sidecar=%s",
        wm.id,
        body.model_id,
        sidecar_written,
    )
    wm_dict["sidecar_written"] = sidecar_written
    return wm_dict


@router.post("/watermark/verify")
async def verify_watermark(
    body: WatermarkVerifyRequest,
    session: SessionDep,
    store: StoreDep,
):
    secret = _resolve_wm_secret()
    # #1: defense-in-depth. Two sources of truth now exist: the sidecar that
    # travels with the model files (verifies a copied model with no Hub DB) and
    # the Hub DB row. When BOTH are present they must AGREE — a tamper in either
    # is a forgery signal, so verified = sidecar_ok AND db_ok. When only one is
    # present (copied model / legacy embed), that source alone decides.
    version_label = await _resolve_version_label(session, body.model_id, body.version_id)
    sidecar_ok: bool | None = None
    sidecar_dict: dict | None = None
    try:
        sidecar_bytes = store.read_sidecar(body.model_id, version_label, _SIDECAR_NAME)
    except NotImplementedError:
        sidecar_bytes = None
        logger.warning("Sidecar read not supported by storage backend: model=%s", body.model_id)
    except Exception:
        sidecar_bytes = None
        logger.exception("Failed to read watermark sidecar: model=%s version=%s", body.model_id, body.version_id)

    if sidecar_bytes:
        try:
            sc = json.loads(sidecar_bytes)
        except json.JSONDecodeError:
            logger.warning("Corrupt watermark sidecar: model=%s version=%s", body.model_id, body.version_id)
            sc = None
        if sc:
            expected_sig = _sign_payload(sc.get("payload", {}), body.model_id, body.version_id, secret)
            sidecar_ok = hmac.compare_digest(sc.get("signature", ""), expected_sig)
            sidecar_dict = sc

    wms = await crud.list_watermarks(session, model_id=body.model_id, version_id=body.version_id)
    db_ok: bool | None = None
    wm_row = None
    if wms:
        wm_row = wms[0]
        payload = json.loads(wm_row.payload) if wm_row.payload else {}
        expected_sig = _sign_payload(payload, body.model_id, body.version_id, secret)
        # E-S6: constant-time compare so a signature mismatch does not leak how
        # many leading bytes matched (timing oracle for forgery).
        db_ok = hmac.compare_digest(wm_row.signature or "", expected_sig)

    # Neither source present.
    if sidecar_ok is None and db_ok is None:
        return {"verified": False, "reason": "No watermark found", "source": "none"}

    # Single-source paths.
    if sidecar_ok is None:
        return {
            "verified": db_ok,
            "source": "database",
            "watermark": _wm_to_dict(wm_row) if db_ok else None,
            "reason": "" if db_ok else "Signature mismatch",
        }
    if db_ok is None:
        return {
            "verified": sidecar_ok,
            "source": "sidecar",
            "watermark": sidecar_dict if sidecar_ok else None,
            "reason": "" if sidecar_ok else "Signature mismatch (sidecar)",
        }

    # Both present — defense in depth: both must verify.
    verified = sidecar_ok and db_ok
    reason = ""
    if not verified:
        bad = []
        if not sidecar_ok:
            bad.append("sidecar")
        if not db_ok:
            bad.append("database")
        reason = "Signature mismatch: " + "+".join(bad)
    return {
        "verified": verified,
        "source": "sidecar+database",
        "watermark": (sidecar_dict if sidecar_ok else _wm_to_dict(wm_row) if db_ok else None) if verified else None,
        "reason": reason,
    }


@router.get("/watermark/list")
async def list_watermarks(session: SessionDep, model_id: str = "", version_id: str = ""):
    wms = await crud.list_watermarks(session, model_id=model_id, version_id=version_id)
    return {"items": [_wm_to_dict(w) for w in wms]}
