import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..db.database import get_engine, init_db
from . import backup, metrics
from .auth import auth_middleware, set_auth_enabled
from .config import Settings
from .deps import init_deps
from .routers import (
    adapt,
    analyze,
    approvals,
    auth,
    benchmarks,
    branches,
    cache,
    cluster,
    deployments,
    downloads,
    encryption,
    evaluations,
    favorites,
    gitlfs,
    hardware,
    inference,
    models,
    monitor,
    quantize,
    quantize_presets,
    ratings,
    recommend,
    security,
    sync,
    system,
    tenants,
    versions,
    watermark,
    webhooks,
)

logger = logging.getLogger(__name__)


async def _reconcile_orphaned_tasks() -> None:
    from ..db.crud import list_quantize_tasks, update_quantize_task
    from ..db.models import TaskStatus
    from .deps import get_session_factory
    from .tasks import resume_quantize

    sf = get_session_factory()
    failed = 0
    restarted = 0
    async with sf() as session:
        running_tasks, _ = await list_quantize_tasks(session, status=TaskStatus.RUNNING.value, page_size=200)
        for t in running_tasks:
            await update_quantize_task(
                session, t.id, status=TaskStatus.FAILED.value,
                error_message="Task orphaned by server restart",
            )
            failed += 1
        pending_tasks, _ = await list_quantize_tasks(session, status=TaskStatus.PENDING.value, page_size=200)
        # E-D2: LoRA merge tasks have no resume path (the MLX merge may have
        # partially fused weights — resuming risks a half-merged version). Fail
        # both RUNNING and PENDING lora merges orphaned by restart so they do
        # not stay stuck RUNNING forever (the prior reconcile pass only handled
        # QuantizeTask, leaving LoraMergeTask silently orphaned).
        from ..db.crud import list_lora_merge_tasks, update_lora_merge_task
        lora_running, _ = await list_lora_merge_tasks(session, status=TaskStatus.RUNNING.value, page_size=200)
        for t in lora_running:
            await update_lora_merge_task(
                session, t.id, status=TaskStatus.FAILED.value,
                error_message="LoRA merge orphaned by server restart",
            )
            failed += 1
        lora_pending, _ = await list_lora_merge_tasks(session, status=TaskStatus.PENDING.value, page_size=200)
        for t in lora_pending:
            await update_lora_merge_task(
                session, t.id, status=TaskStatus.FAILED.value,
                error_message="LoRA merge not resumed after restart (no safe resume path)",
            )
            failed += 1
        # R4: DistributedTask (cluster fan-out quantize/load) has no safe local
        # resume path — the work happens on remote nodes whose coordinator state
        # is opaque to the hub. A restart leaves any RUNNING/PENDING distributed
        # task stuck forever (the prior reconcile pass never touched
        # DistributedTask). Fail both so the operator sees them and resubmits.
        from sqlalchemy import select

        from ..db.models import DistributedTask, DistributedTaskStatus
        for st in (DistributedTaskStatus.RUNNING, DistributedTaskStatus.PENDING):
            rows = (await session.execute(
                select(DistributedTask).where(DistributedTask.status == st)
            )).scalars().all()
            for t in rows:
                t.status = DistributedTaskStatus.FAILED
                t.completed_at = None
                failed += 1
            if rows:
                logger.warning("Failed %d orphaned distributed task(s) in %s", len(rows), st.value)
        await session.commit()
    for t in pending_tasks:
        try:
            resumed = await resume_quantize(
                task_id=t.id,
                source_version_id=t.source_version_id,
                target_format=t.target_format,
                quant_bits=t.quant_bits,
            )
            if resumed:
                restarted += 1
        except Exception:
            logger.exception("Failed to restart pending task: id=%s", t.id)
    if failed or restarted:
        logger.warning("Startup task recovery: %d orphaned failed, %d pending resumed", failed, restarted)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = get_engine(
            settings.db_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )
        # P1-19: a failing init_db (corrupt DB, disk full, bad DDL) used to crash
        # startup hard, leaving the operator with no API to inspect/recover. Now
        # log loudly and degrade — metadata reads may partially work or fail per
        # request, but the process stays up so /system/health and logs are reachable.
        try:
            await init_db(engine)
        except Exception:
            logger.exception("init_db failed — starting in degraded mode; DB ops will error until recovered")
        init_deps(settings, engine)
        set_auth_enabled(settings.auth_enabled)
        await _reconcile_orphaned_tasks()
        # H10: probe Fusion-MLX version compatibility at startup. Best-effort
        # (MLX may start after the hub) — a mismatch or unreachable MLX is
        # logged loudly but does not block startup, so the hub still serves
        # metadata while the operator fixes the base.
        try:
            from ..api.base_binding import FusionMLXBase
            base = FusionMLXBase(mlx_url=settings.mlx_url)
            compat = await base.check_compatibility(">=0.5.0")
            if compat.get("compatible"):
                logger.info("Fusion-MLX compatible: %s", compat)
            else:
                logger.warning(
                    "Fusion-MLX compatibility check: %s — inference/quantize/convert "
                    "will fail until the base is upgraded or started",
                    compat,
                )
        except Exception:
            logger.warning("Fusion-MLX compatibility probe failed at startup", exc_info=True)
        backup.start_backup_scheduler()
        logger.info("Fusion Model Hub started: data_dir=%s auth=%s", settings.data_dir, settings.auth_enabled)
        yield
        backup.stop_backup_scheduler()
        # H8: close the shared httpx transports pooled for the MLX hot path.
        try:
            from .http_client import close_all_transports
            await close_all_transports()
        except Exception:
            logger.warning("Failed to close pooled httpx transports", exc_info=True)
        # Dispose all async engines so aiosqlite worker threads shut down before
        # the event loop closes (prevents `RuntimeError: Event loop is closed`).
        try:
            from ..db.database import dispose_all_engines
            await dispose_all_engines()
        except Exception:
            logger.warning("Failed to dispose async engines", exc_info=True)

    app = FastAPI(
        title="Fusion Model Hub",
        description="Unified model repository & manager for the Fusion-MLX ecosystem",
        version=__version__,
        lifespan=lifespan,
    )

    cors_allow_credentials = "*" not in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        import re as _re
        import uuid as _uuid

        from fastapi import HTTPException as FastAPIHTTPException
        if isinstance(exc, FastAPIHTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        # E-E13: exc.__repr__/str(exc) routinely embeds the SQLAlchemy db_url
        # (with creds) and absolute filesystem paths from the request — logging
        # that verbatim at ERROR leaks secrets/paths into log aggregators.
        # Keep only the exception class name + a per-incident trace_id in the
        # ERROR line; stash the full repr at DEBUG under the same trace_id so
        # an operator with DEBUG access can still correlate, while production
        # ERROR logs stay clean. The response carries the trace_id so a user
        # report can be matched to the log line without exposing internals.
        trace_id = _uuid.uuid4().hex[:12]
        exc_repr = repr(exc)
        redacted = _re.sub(
            r"(://[^:\s]+:[^\s@]+@|/[^\s]*\.(?:db|sqlite|sqlite3|json))",
            "[REDACTED]",
            exc_repr,
        )
        logger.error(
            "Unhandled exception: %s %s -> %s trace_id=%s",
            request.method, request.url.path, type(exc).__name__, trace_id,
        )
        logger.debug("Unhandled exception detail trace_id=%s: %s", trace_id, redacted)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "trace_id": trace_id},
        )

    # E-D1: illegal model/version status transitions are a client error (409),
    # not a 500. Map before the generic handler so callers get a clear conflict.
    from ..db.crud import InvalidTransition

    @app.exception_handler(InvalidTransition)
    async def invalid_transition_handler(request: Request, exc: InvalidTransition):
        logger.info("Status transition rejected: %s %s -> %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(versions.router, prefix="/api/v1")
    app.include_router(quantize.router, prefix="/api/v1")
    app.include_router(inference.router, prefix="/api/v1")
    app.include_router(monitor.router, prefix="/api/v1")
    app.include_router(quantize_presets.router, prefix="/api/v1")
    app.include_router(system.router, prefix="/api/v1")
    app.include_router(cluster.router, prefix="/api/v1")
    app.include_router(tenants.router, prefix="/api/v1")
    app.include_router(webhooks.router, prefix="/api/v1")
    app.include_router(deployments.router, prefix="/api/v1")
    app.include_router(evaluations.router, prefix="/api/v1")
    app.include_router(sync.router, prefix="/api/v1")
    app.include_router(security.router, prefix="/api/v1")
    app.include_router(watermark.router, prefix="/api/v1")
    app.include_router(encryption.router, prefix="/api/v1")
    app.include_router(approvals.router, prefix="/api/v1")
    app.include_router(gitlfs.router, prefix="/api/v1")
    app.include_router(ratings.router, prefix="/api/v1")
    app.include_router(favorites.router, prefix="/api/v1")
    app.include_router(branches.router, prefix="/api/v1")
    app.include_router(cache.router, prefix="/api/v1")
    app.include_router(downloads.router, prefix="/api/v1")
    app.include_router(hardware.router, prefix="/api/v1")
    app.include_router(recommend.router, prefix="/api/v1")
    app.include_router(adapt.router, prefix="/api/v1")
    app.include_router(benchmarks.router, prefix="/api/v1")
    app.include_router(analyze.router, prefix="/api/v1")

    app.middleware("http")(auth_middleware)
    app.middleware("http")(metrics.metrics_middleware)

    @app.get("/metrics", tags=["system"])
    async def prometheus_metrics():
        # E-S11: do not expose internal telemetry unless the operator explicitly
        # opted in via FMH_EXPOSE_METRICS=true. A bare 404 (not 403) avoids
        # confirming the endpoint exists when disabled.
        if not getattr(settings, "expose_metrics", False):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return metrics.metrics_response()

    return app
