import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

_window_seconds = 60
_buckets: dict[str, list[float]] = defaultdict(list)
_EVICT_THRESHOLD = 4096
_last_evict_check = 0.0


def _maybe_evict_dead_keys(now: float) -> None:
    global _last_evict_check
    if len(_buckets) < _EVICT_THRESHOLD and now - _last_evict_check < _window_seconds:
        return
    cutoff = now - _window_seconds
    dead = [k for k, ts in _buckets.items() if not ts or ts[-1] <= cutoff]
    for k in dead:
        del _buckets[k]
    _last_evict_check = now
    if dead:
        logger.info("Rate limit evicted %d dead key buckets (remaining=%d)", len(dead), len(_buckets))


def check_rate_limit(key_prefix: str, qps_limit: int) -> bool:
    if qps_limit <= 0:
        return True
    now = time.time()
    _maybe_evict_dead_keys(now)
    cutoff = now - _window_seconds
    bucket = _buckets[key_prefix]
    bucket[:] = [t for t in bucket if t > cutoff]
    if len(bucket) >= qps_limit:
        logger.warning("Rate limit exceeded: key_prefix=%s limit=%d", key_prefix, qps_limit)
        return False
    bucket.append(now)
    return True


def reset_rate_limits() -> None:
    _buckets.clear()
