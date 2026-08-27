import logging
import time

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "fmh_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "fmh_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

ACTIVE_REQUESTS = Gauge(
    "fmh_http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "endpoint"],
)

MODELS_TOTAL = Gauge(
    "fmh_models_total",
    "Total number of registered models",
)

VERSIONS_TOTAL = Gauge(
    "fmh_versions_total",
    "Total number of model versions",
)

QUANTIZE_TASKS_RUNNING = Gauge(
    "fmh_quantize_tasks_running",
    "Number of running quantize tasks",
)

MLX_STATUS = Gauge(
    "fmh_mlx_available",
    "Fusion-MLX availability (1=available, 0=unavailable)",
)


def _normalize_path(path: str) -> str:
    parts = path.strip("/").split("/")
    normalized = []
    for p in parts:
        if len(p) > 16 or p.isdigit():
            normalized.append(":id")
        else:
            normalized.append(p)
    return "/" + "/".join(normalized) if normalized else "/"


async def metrics_middleware(request: Request, call_next) -> Response:
    path = _normalize_path(request.url.path)
    if path == "/metrics" or path.startswith(("/docs", "/openapi")):
        return await call_next(request)

    method = request.method
    ACTIVE_REQUESTS.labels(method=method, endpoint=path).inc()
    start = time.monotonic()
    try:
        response = await call_next(request)
        duration = time.monotonic() - start
        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=str(response.status_code)).inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
        return response
    except Exception:
        duration = time.monotonic() - start
        REQUEST_COUNT.labels(method=method, endpoint=path, status_code="500").inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
        raise
    finally:
        ACTIVE_REQUESTS.labels(method=method, endpoint=path).dec()


async def update_resource_metrics(session) -> None:
    from ..db import crud

    try:
        _, model_count = await crud.list_models(session, page_size=1)
        MODELS_TOTAL.set(model_count)
    except Exception:
        logger.debug("Failed to update resource metrics")


def metrics_response():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
