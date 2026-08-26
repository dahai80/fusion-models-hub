from __future__ import annotations

import asyncio
import logging

import pytest

logger = logging.getLogger(__name__)


async def _dispose_engines():
    from fusion_model_hub.db.database import dispose_all_engines

    await dispose_all_engines()


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
