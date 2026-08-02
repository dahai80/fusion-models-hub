import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

_window_seconds = 60
_buckets: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(key_prefix: str, qps_limit: int) -> bool:
    if qps_limit <= 0:
        return True
    now = time.time()
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
