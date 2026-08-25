import asyncio
import hashlib
import hmac
import json
import logging

import httpx

from ..db import crud

logger = logging.getLogger(__name__)

# H11: webhook dispatch is a service-layer concern (called by tasks.py,
# quantize.py, inference hot-reload) but previously lived in the webhooks
# router, inverting the layering — service code depended on a router module.
# Moved here so the service layer owns event delivery; the router re-exports
# dispatch_webhook_event for backward compatibility.

_WEBHOOK_MAX_RETRIES = 3
_WEBHOOK_BACKOFF_BASE = 1.0


def _sign_payload(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


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
    from .deps import get_session_factory
    sf = get_session_factory()
    # E-E4: the outer try/except must NOT wrap the per-webhook for-loop. A prior
    # version put the whole loop inside one try, so a single webhook raising an
    # exception that escaped _send_webhook_with_retry (e.g. an httpx constructor
    # SSL error, or an asyncio.CancelledError out of the retry backoff sleep)
    # was caught here and the function returned — every subscriber AFTER the
    # failing one was silently skipped, including a canary "deploy" webhook
    # triggered by the same event. Build the payload + list first (those can
    # still fail and abort the whole dispatch), then dispatch each webhook in
    # its own try so one bad URL never blocks the rest.
    try:
        async with sf() as session:
            webhooks = await crud.list_webhooks(session, tenant_id=tenant_id)
        payload_bytes = json.dumps({"event": event, "data": data}).encode()
    except Exception:
        logger.exception("dispatch_webhook_event could not load webhooks for event=%s", event)
        return

    for w in webhooks:
        if not w.is_active:
            continue
        if w.events:
            subscribed = {e.strip() for e in w.events.split(",") if e.strip()}
            if event not in subscribed:
                continue
        signature = _sign_payload(payload_bytes, w.secret) if w.secret else ""
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": event,
            "X-Webhook-Signature": signature,
        }
        try:
            await _send_webhook_with_retry(w.url, payload_bytes, headers, w.id, event)
        except Exception:
            # E-E4: isolate per-webhook failures. Log and continue to the next
            # subscriber so one broken URL cannot suppress delivery to the rest.
            logger.exception(
                "dispatch_webhook_event: subscriber id=%s url=%s raised, skipping (event=%s)",
                w.id, w.url, event,
            )
