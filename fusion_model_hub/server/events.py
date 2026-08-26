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

# P1-16: bound concurrent in-flight webhook deliveries so a burst of events
# cannot spawn an unbounded number of tasks (each holding an httpx client).
_WEBHOOK_CONCURRENCY = 32
_webhook_semaphore: asyncio.Semaphore | None = None
# P1-16: keep strong refs to fire-and-forget tasks so they are not GC'd mid-
# flight. Cleared as each completes via done-callback.
_pending_webhook_tasks: set[asyncio.Task] = set()


def _get_webhook_semaphore() -> asyncio.Semaphore:
    global _webhook_semaphore
    if _webhook_semaphore is None:
        _webhook_semaphore = asyncio.Semaphore(_WEBHOOK_CONCURRENCY)
    return _webhook_semaphore


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
        # P1-16: fire-and-forget. Before, dispatch_webhook_event awaited
        # _send_webhook_with_retry inline — up to 3 retries x (5s timeout +
        # exponential backoff) ~= 21s of blocking per subscriber, paid by the
        # caller (version publish, quantize complete, hot-reload). The caller
        # only needs "attempted", not the delivery result (retries + give-up are
        # logged inside _send_webhook_with_retry). Spawn a bounded task per
        # subscriber and return immediately; one slow/dead URL no longer stalls
        # the request path or the subscribers behind it.
        task = asyncio.create_task(
            _deliver_one_webhook(w.url, payload_bytes, headers, w.id, event),
            name=f"webhook-{w.id}-{event}",
        )
        _pending_webhook_tasks.add(task)
        task.add_done_callback(_pending_webhook_tasks.discard)


async def _deliver_one_webhook(
    url: str, payload_bytes: bytes, headers: dict, webhook_id: str, event: str,
) -> None:
    # P1-16: per-subscriber delivery coroutine, gated by the concurrency
    # semaphore so a burst does not open dozens of httpx clients at once.
    # E-E4: isolate per-webhook failures — one broken URL must not suppress
    # the rest (they run as independent tasks now, but the guard is kept).
    sem = _get_webhook_semaphore()
    try:
        async with sem:
            await _send_webhook_with_retry(url, payload_bytes, headers, webhook_id, event)
    except Exception:
        logger.exception(
            "dispatch_webhook_event: subscriber id=%s url=%s raised, skipping (event=%s)",
            webhook_id, url, event,
        )
