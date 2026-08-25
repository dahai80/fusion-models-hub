"""H8: process-pooled httpx client for high-frequency MLX calls.

The inference router issues many short-lived async HTTP requests to Fusion-MLX
(serve / chat / unload / serve-status). Each ``async with httpx.AsyncClient``
opens and tears down a fresh connection pool, so under load the Hub spends
real time in TLS/TCP setup to a loopback peer. This module keeps ONE
``AsyncHTTPTransport`` per base_url alive for the process and hands out thin
``AsyncClient`` wrappers whose ``aclose`` is a no-op (the shared transport is
NOT closed when a wrapper exits its ``async with`` block). Connections are
therefore reused across calls.

Design constraints:
- ``PoolClient`` subclasses ``httpx.AsyncClient``, so ``async with
  PoolClient(...) as c`` and ``await c.post(...)`` are unchanged.
- Tests that do ``patch("...inference.httpx.AsyncClient", return_value=mock)``
  still intercept, because ``inference`` imports this module as ``httpx`` and
  reads ``httpx.AsyncClient`` (== ``PoolClient``); the mock supplies its own
  ``__aenter__`` / ``__aexit__`` / ``post``.
- Per-call ``timeout`` passes straight through to the wrapper constructor and
  does not touch the shared transport, so serve (60s) and chat (120s) keep
  their own deadlines.
- Low-frequency / external-host callers (sync registry, HF search, downloads,
  convert, webhooks, SDK) keep using real ``httpx`` directly — only the MLX
  hot path opts into pooling.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

# Re-export the httpx exception/response types that routers reference via the
# aliased module (``from . import http_client as httpx`` then ``httpx.ConnectError``).
ConnectError = httpx.ConnectError
ConnectTimeout = httpx.ConnectTimeout
ReadTimeout = httpx.ReadTimeout
TimeoutException = httpx.TimeoutException
RequestError = httpx.RequestError
HTTPStatusError = httpx.HTTPStatusError
HTTPError = httpx.HTTPError
Response = httpx.Response

# Alias so call sites read ``httpx.AsyncClient(...)`` and get the pooled impl.
# Set AFTER the class is defined below — assigned at the bottom of the module.

# One persistent transport per base_url (or "default" when none). The transport
# owns the actual connection pool; wrappers reuse it. Keyed by the base_url the
# wrapper was constructed with so different MLX hosts do not collide.
_TRANSPORT_POOL: dict[str, httpx.AsyncHTTPTransport] = {}


def _get_shared_transport(base_url: str) -> httpx.AsyncHTTPTransport:
    key = base_url or "default"
    transport = _TRANSPORT_POOL.get(key)
    if transport is None:
        transport = httpx.AsyncHTTPTransport()
        _TRANSPORT_POOL[key] = transport
        logger.info("http_client: created shared transport for base_url=%s", key or "<inline>")
    return transport


class PoolClient(httpx.AsyncClient):
    """httpx.AsyncClient whose aclose() does NOT close its shared transport.

    Identical constructor signature to httpx.AsyncClient, so call sites are
    unchanged (``async with httpx.AsyncClient(timeout=60.0) as client``).
    """

    def __init__(self, *args, base_url: str | None = None, **kwargs):
        # Key the shared transport by base_url (empty string when None) so all
        # call sites that build full URLs inline (base_url omitted) share one
        # pool. Do NOT forward base_url=None to httpx — its URL() rejects None;
        # only pass it through when a real value was given.
        transport = _get_shared_transport(str(base_url) if base_url else "")
        kwargs["transport"] = transport
        if base_url is not None:
            kwargs["base_url"] = base_url
        super().__init__(*args, **kwargs)

    async def aclose(self) -> None:
        # Do NOT close the shared transport — other wrappers may reuse it.
        # Only flip the client state so __aexit__ sees a closed client.
        if self._state != httpx._client.ClientState.CLOSED:
            self._state = httpx._client.ClientState.CLOSED


async def close_all_transports() -> None:
    """Shutdown hook: close every pooled transport (app lifespan / tests)."""
    global _TRANSPORT_POOL
    for key, transport in _TRANSPORT_POOL.items():
        try:
            await transport.aclose()
            logger.info("http_client: closed shared transport base_url=%s", key or "<inline>")
        except Exception:
            logger.warning("http_client: failed closing transport base_url=%s", key, exc_info=True)
    _TRANSPORT_POOL = {}


# Routers import this module as ``httpx`` and read ``httpx.AsyncClient``; route
# those to the pooled subclass. Defined here (after PoolClient) so the alias
# resolves to the real class, not a forward reference.
AsyncClient = PoolClient
