import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..db.database import get_engine, init_db
from .config import Settings
from .deps import init_deps
from .auth import auth_middleware, set_auth_enabled
from .routers import auth, cluster, inference, models, quantize, system, versions

logger = logging.getLogger(__name__)


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
        logger.info("Fusion Model Hub started: data_dir=%s auth=%s", settings.data_dir, settings.auth_enabled)
        yield

    app = FastAPI(
        title="Fusion Model Hub",
        description="Unified model repository & manager for the Fusion-MLX ecosystem",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
        logger.error("Unhandled exception: %s %s -> %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")
    app.include_router(versions.router, prefix="/api/v1")
    app.include_router(quantize.router, prefix="/api/v1")
    app.include_router(inference.router, prefix="/api/v1")
    app.include_router(system.router, prefix="/api/v1")
    app.include_router(cluster.router, prefix="/api/v1")

    app.middleware("http")(auth_middleware)

    return app
