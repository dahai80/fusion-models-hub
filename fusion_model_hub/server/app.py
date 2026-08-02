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
    cluster,
    deployments,
    encryption,
    evaluations,
    favorites,
    gitlfs,
    hardware,
    inference,
    models,
    quantize,
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
    from .tasks import submit_quantize

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
    for t in pending_tasks:
        try:
            await submit_quantize(
                source_version_id=t.source_version_id,
                target_format=t.target_format,
                quant_bits=t.quant_bits,
                calibration_dataset=t.calibration_dataset,
            )
            restarted += 1
        except Exception:
            logger.exception("Failed to restart pending task: id=%s", t.id)
    if failed or restarted:
        logger.warning("Startup task recovery: %d orphaned failed, %d pending restarted", failed, restarted)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        set_auth_enabled(settings.auth_enabled)
        await _reconcile_orphaned_tasks()
        backup.start_backup_scheduler()
        logger.info("Fusion Model Hub started: data_dir=%s auth=%s", settings.data_dir, settings.auth_enabled)
        yield
        backup.stop_backup_scheduler()

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
        from fastapi import HTTPException as FastAPIHTTPException
        if isinstance(exc, FastAPIHTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.error("Unhandled exception: %s %s -> %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(versions.router, prefix="/api/v1")
    app.include_router(quantize.router, prefix="/api/v1")
    app.include_router(inference.router, prefix="/api/v1")
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
    app.include_router(hardware.router, prefix="/api/v1")
    app.include_router(recommend.router, prefix="/api/v1")
    app.include_router(adapt.router, prefix="/api/v1")
    app.include_router(benchmarks.router, prefix="/api/v1")
    app.include_router(analyze.router, prefix="/api/v1")

    app.middleware("http")(auth_middleware)
    app.middleware("http")(metrics.metrics_middleware)

    @app.get("/metrics", tags=["system"])
    async def prometheus_metrics():
        return metrics.metrics_response()

    return app
