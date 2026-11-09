import asyncio
import contextlib
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Issue #96: /system/health used to run the MLX probe serially per request;
# a stalled dependency pushed the handler past the supervisor's probe timeout
# and the service flapped down even though the process was alive. The probe
# now runs in a background poller and the handler serves a cached snapshot;
# `?deep=1` (or a missing snapshot, e.g. in tests without a lifespan) falls
# back to the per-request probe.

_PROBE_TIMEOUT = 3.0

_snapshot: dict = {}
_poller_task: asyncio.Task | None = None


async def probe_mlx(settings, probe_timeout: float = _PROBE_TIMEOUT) -> dict:
    # #58: mlx_url defaults to the gateway — /health is public there but send
    # the gateway key for consistency with the rest of the LLM surface.
    # #71: probe the lightweight /healthz first (no backend fan-out, no
    # recursion gateway /health -> hub /health -> gateway /health). 404 falls
    # back to /health for targets without a /healthz (direct fusion-mlx).
    mlx_headers = {"X-Fusion-Source": "model-hub"}
    if settings.mlx_gateway_api_key:
        mlx_headers["Authorization"] = f"Bearer {settings.mlx_gateway_api_key}"
    try:
        async with httpx.AsyncClient(timeout=probe_timeout) as client:
            resp = await client.get(f"{settings.mlx_url}/healthz", headers=mlx_headers)
            if resp.status_code == 404:
                resp = await client.get(f"{settings.mlx_url}/health", headers=mlx_headers)
            if resp.status_code == 200:
                try:
                    info = resp.json()
                except Exception:
                    info = {"raw": resp.text[:200]}
                return {"status": "available", "info": info}
            return {"status": f"error_{resp.status_code}", "info": {}}
    except httpx.ConnectError:
        return {"status": "offline", "info": {}}
    except Exception as e:
        logger.warning("MLX health check failed: %s", e)
        return {"status": "error", "info": {}}


def get_snapshot() -> dict | None:
    return dict(_snapshot) if _snapshot else None


def set_snapshot(probe: dict) -> None:
    _snapshot.clear()
    _snapshot.update(probe)
    _snapshot["checked_at"] = time.time()


async def _poll_loop(settings, interval: float) -> None:
    while True:
        probe = await probe_mlx(settings)
        set_snapshot(probe)
        await asyncio.sleep(interval)


def start_health_poller(settings) -> asyncio.Task | None:
    global _poller_task
    interval = getattr(settings, "health_poll_seconds", 10)
    if interval <= 0:
        logger.info("Health poller disabled (health_poll_seconds <= 0)")
        return None
    if _poller_task is not None and not _poller_task.done():
        return _poller_task
    _poller_task = asyncio.create_task(_poll_loop(settings, interval))
    logger.info("Health poller started: interval=%ss target=%s", interval, settings.mlx_url)
    return _poller_task


async def stop_health_poller() -> None:
    global _poller_task
    if _poller_task is None:
        return
    _poller_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _poller_task
    _poller_task = None
    logger.info("Health poller stopped")
