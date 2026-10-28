from __future__ import annotations

import asyncio
import logging

import pytest

logger = logging.getLogger(__name__)


async def _dispose_engines():
    from fusion_model_hub.db.database import dispose_all_engines

    await dispose_all_engines()


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    # The bootstrap/API-key rate limiter is module-global in-memory state
    # (server/rate_limit.py). Without a reset, tests that hit /api/v1/auth/keys
    # bootstrap within the same 60s window trip the limiter for later tests
    # (flaky 429 in test-order-dependent runs). Reset before and after each test.
    from fusion_model_hub.server.rate_limit import reset_rate_limits

    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.fixture(autouse=True)
async def _dispose_engines_after_async_test():
    yield
    try:
        await _dispose_engines()
    except Exception:
        logger.warning("async engine teardown failed", exc_info=True)


@pytest.fixture(autouse=True)
def _dispose_engines_after_sync_test(request):
    yield
    if request.node.get_closest_marker("asyncio") is not None:
        return
    try:
        asyncio.run(_dispose_engines())
    except Exception:
        logger.warning("sync engine teardown failed", exc_info=True)
