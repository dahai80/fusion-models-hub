import asyncio
import hashlib
import hmac
import json
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WebhookCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    url: str = Field(..., min_length=1, max_length=512)
    secret: str = Field("", max_length=128)
    events: str = Field("model.created,model.deleted", max_length=512)


class WebhookOut(BaseModel):
    id: str
    name: str
    url: str
    secret: str
    events: str
    is_active: bool
    model_config = {"from_attributes": True}


@router.post("", status_code=201, response_model=WebhookOut)
async def create_webhook(body: WebhookCreate, session: SessionDep):
    w = await crud.create_webhook(
        session, name=body.name, url=body.url, secret=body.secret, events=body.events,
    )
    return w


@router.get("")
async def list_webhooks(session: SessionDep, request: Request):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    items = await crud.list_webhooks(session, tenant_id=tenant_id)
    logger.info("Listed webhooks: %d (tenant=%s)", len(items), tenant_id)
    return {"webhooks": items, "total": len(items)}


@router.get("/{webhook_id}", response_model=WebhookOut)
async def get_webhook(webhook_id: str, session: SessionDep):
    w = await crud.get_webhook(session, webhook_id)
    if not w:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return w


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str, session: SessionDep):
    ok = await crud.delete_webhook(session, webhook_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"detail": "deleted"}


def _sign_payload(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


_WEBHOOK_MAX_RETRIES = 3
_WEBHOOK_BACKOFF_BASE = 1.0


async def _send_webhook_with_retry(
    url: str, payload_bytes: bytes, headers: dict, webhook_id: str, event: str,
) -> None:
    for attempt in range(1, _WEBHOOK_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, content=payload_bytes, headers=headers)
                if resp.status_code < 500:
                    logger.info(
                        "Webhook dispatched: id=%s event=%s status=%d attempt=%d",
                        webhook_id, event, resp.status_code, attempt,
                    )
                    return
                logger.warning(
                    "Webhook server error: id=%s status=%d attempt=%d",
                    webhook_id, resp.status_code, attempt,
                )
        except Exception:
            logger.warning("Webhook dispatch failed: id=%s url=%s attempt=%d", webhook_id, url, attempt)
        if attempt < _WEBHOOK_MAX_RETRIES:
            delay = _WEBHOOK_BACKOFF_BASE * (2 ** (attempt - 1))
            await asyncio.sleep(delay)
    logger.error(
        "Webhook gave up after %d retries: id=%s url=%s event=%s",
        _WEBHOOK_MAX_RETRIES, webhook_id, url, event,
    )


async def dispatch_webhook_event(event: str, data: dict, tenant_id: str = "") -> None:
    from ..deps import get_session_factory
    sf = get_session_factory()
    try:
        async with sf() as session:
            webhooks = await crud.list_webhooks(session, tenant_id=tenant_id)
        payload_bytes = json.dumps({"event": event, "data": data}).encode()
        for w in webhooks:
            if not w.is_active:
                continue
            if w.events and event not in w.events:
                continue
            signature = _sign_payload(payload_bytes, w.secret) if w.secret else ""
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Event": event,
                "X-Webhook-Signature": signature,
            }
            await _send_webhook_with_retry(w.url, payload_bytes, headers, w.id, event)
    except Exception:
        logger.exception("dispatch_webhook_event error for event=%s", event)
