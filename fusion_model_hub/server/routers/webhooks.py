import logging

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


def _caller_tenant(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or ""


@router.post("", status_code=201, response_model=WebhookOut)
async def create_webhook(body: WebhookCreate, session: SessionDep, request: Request):
    from ..ssrf import validate_external_url
    validate_external_url(body.url)
    tenant_id = _caller_tenant(request)
    w = await crud.create_webhook(
        session, name=body.name, url=body.url, secret=body.secret,
        events=body.events, tenant_id=tenant_id,
    )
    logger.info("Created webhook: id=%s tenant=%s", w.id, tenant_id)
    return w


@router.get("")
async def list_webhooks(session: SessionDep, request: Request):
    tenant_id = _caller_tenant(request)
    items = await crud.list_webhooks(session, tenant_id=tenant_id)
    logger.info("Listed webhooks: %d (tenant=%s)", len(items), tenant_id)
    return {"webhooks": items, "total": len(items)}


@router.get("/{webhook_id}", response_model=WebhookOut)
async def get_webhook(webhook_id: str, session: SessionDep, request: Request):
    w = await crud.get_webhook(session, webhook_id)
    if not w:
        raise HTTPException(status_code=404, detail="Webhook not found")
    tenant_id = _caller_tenant(request)
    if tenant_id and w.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return w


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str, session: SessionDep, request: Request):
    w = await crud.get_webhook(session, webhook_id)
    if not w:
        raise HTTPException(status_code=404, detail="Webhook not found")
    tenant_id = _caller_tenant(request)
    if tenant_id and w.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Webhook not found")
    await crud.delete_webhook(session, webhook_id)
    return {"detail": "deleted"}


# H11: webhook delivery (_sign_payload, _send_webhook_with_retry,
# dispatch_webhook_event) moved to the service layer (server/events.py).
# Re-exported here so existing `from .routers.webhooks import
# dispatch_webhook_event` call sites (tasks.py, quantize.py) keep working
# without a sweeping import rewrite.
from ..events import (  # noqa: F401
    _send_webhook_with_retry,
    _sign_payload,
    dispatch_webhook_event,
)
