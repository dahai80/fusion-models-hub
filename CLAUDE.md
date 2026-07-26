# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion-Model-Hub is the unified model repository and management center for the Fusion-MLX ecosystem on macOS Apple Silicon. It provides model discovery → download → conversion → version management → inference service across the Fusion ecosystem (Fusion-Desktop, Fusion-Agent Studio, Fusion-KB, Fusion-Bench).

**Critical constraint:** All inference, conversion, and verification is 100% delegated to the Fusion-MLX base via HTTP API (`http://localhost:11434`). This project never imports `mlx`, `mlx-lm`, `torch`, or `transformers` directly.

## Development Commands

```bash
# Activate environment (always run first)
source .venv/bin/activate

# Run all tests
pytest

# Run API integration tests
pytest tests/test_api.py -v

# Run a single test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=fusion_model_hub --cov-report=term-missing

# Install in editable mode
pip install -e ".[test]"

# Start API server
fusion-model-hub --host 0.0.0.0 --port 8080
```

Fusion-MLX lifecycle (for integration tests requiring a real model server):
```bash
~/claude-home/fusion-mlx/start.sh start   # Start Fusion-MLX server
~/claude-home/fusion-mlx/start.sh stop    # Stop Fusion-MLX server
```

## Architecture

```
fusion_model_hub/
├── db/                          # Database layer (Phase 1: SQLite + aiosqlite)
│   ├── models.py                # SQLAlchemy ORM: Model, ModelVersion, ModelTag, QuantizeTask, ApiKey, AuditLog, ClusterNode
│   ├── database.py              # Async engine & session factory
│   └── crud.py                  # Async CRUD operations with field whitelists
├── storage/
│   └── local_store.py           # File storage: chunked upload, SHA256, assemble
├── server/                      # FastAPI REST API server
│   ├── app.py                   # App factory with lifespan + middleware
│   ├── config.py                # Settings dataclass (env vars: FMH_DATA_DIR, FMH_MLX_URL, FMH_AUTH_ENABLED, FMH_CORS_ORIGINS, FMH_MAX_UPLOAD_SIZE_MB)
│   ├── deps.py                  # Dependency injection (SessionDep, StoreDep, SettingsDep)
│   ├── auth.py                  # Auth middleware: API key validation, public path whitelist, audit logging
│   ├── tasks.py                 # Async task runner for quantize operations (asyncio.create_task)
│   ├── __main__.py              # CLI entry point (uvicorn)
│   └── routers/
│       ├── models.py            # /api/v1/models CRUD + HF import + sync + batch ops + compare
│       ├── versions.py          # /api/v1/versions upload/download/status/benchmark/rollback/deprecate/retire + URL download (SSRF-protected)
│       ├── system.py            # /api/v1/system health + storage stats + audit log
│       ├── auth.py              # /api/v1/auth API key management (create/list/deactivate/delete)
│       ├── inference.py         # /api/v1/inference proxy (chat/completions/embeddings) with loaded-model tracking + TTL eviction
│       ├── quantize.py          # /api/v1/quantize submit/monitor/list running tasks
│       └── cluster.py           # /api/v1/cluster node management + heartbeat
├── api/
│   └── base_binding.py          # FusionMLXBase — detects/verifies Fusion-MLX availability via HTTP
├── convert/
│   └── converter.py             # ModelConverter — converts HF/PyTorch/GGUF → MLX via Fusion-MLX
├── manage/
│   └── manager.py               # LocalModelManager — register/list/activate/delete for locally installed models
└── repo/
    ├── models.py                # Data models: ModelInfo, DownloadTask, enums
    ├── registry.py              # ModelRegistry — in-memory catalog with filtering/search
    └── downloader.py            # ModelDownloader — async download with HTTP Range resume + SHA256
```

### Key Design Decisions

- **FastAPI lifespan:** App initialization uses `asynccontextmanager` lifespan (not deprecated `on_event`). Tests manually call `init_deps()` since httpx ASGITransport doesn't trigger ASGI lifespan.
- **Form fields with UploadFile:** Version upload endpoints use explicit `Form()` annotations for all fields alongside `UploadFile`, ensuring multipart form data parsing works correctly.
- **All async I/O:** Database (aiosqlite), storage, downloads, and Fusion-MLX communication all use async. Tests use `pytest-asyncio`.
- **Dependency injection:** `SessionDep`, `StoreDep`, `SettingsDep` are `Annotated` types using FastAPI `Depends`. Module-level singletons initialized via `init_deps()`.
- **Chunked upload:** Large files split into 5MB chunks, assembled server-side with SHA256 verification.
- **Version lifecycle:** draft → testing → published → deprecated → retired state machine.
- **SQLite → PostgreSQL migration path:** All DB access through async SQLAlchemy ORM; swapping to PostgreSQL only requires changing `db_url`.
- **Field whitelists:** `update_model`, `update_version`, `update_quantize_task` use explicit allowlist sets (`_MODEL_UPDATABLE`, `_VERSION_UPDATABLE`, `_TASK_UPDATABLE`) instead of `hasattr` to prevent accidental field overwrites.
- **Auth middleware:** Opt-in via `Settings.auth_enabled`. Write methods (POST/PUT/DELETE/PATCH) require `X-API-Key` header when enabled. `PUBLIC_PATHS` exempt specific endpoints. Audit logs recorded for all write operations.
- **Async task management:** Quantize tasks use `asyncio.create_task` with `_running_tasks` dict tracking. Tasks run independently and update status via DB on completion/failure.
- **Inference proxy pattern:** `_loaded_models` in-memory dict tracks served models. TTL eviction (1 hour) calls MLX unload before removing entries. All inference requests proxy to Fusion-MLX with model name substitution.
- **Cluster heartbeat:** Cluster nodes report liveness via heartbeat endpoint; stale nodes can be identified by `last_heartbeat` timestamp.
- **SSRF prevention:** `sync_registry` source_url and `UrlDownloadRequest.url` validate scheme (http/https only) and block internal IPs (localhost, 10.x, 172.16-31.x, 192.168.x, 169.254.169.254).
- **Download resume:** `ModelDownloader` preserves `.part` files on failure and sends HTTP Range headers to resume from last byte on retry.

## Dependencies

- `fastapi>=0.115.0` — async REST framework
- `uvicorn[standard]>=0.30.0` — ASGI server
- `sqlalchemy[asyncio]>=2.0.0` — async ORM
- `aiosqlite>=0.20.0` — async SQLite driver
- `python-multipart>=0.0.9` — form data parsing
- `httpx>=0.27.0` — async HTTP client
- `pydantic>=2.0.0` — data validation
- Python >=3.12

## Test Structure

- `tests/test_api.py` — API integration tests (FastAPI TestClient with in-memory SQLite). 97 tests covering all routers.
- `tests/test_core.py` — unit tests for repo/convert/manage/api modules
- `tests/test_coverage.py` — additional coverage tests

API tests use `httpx.AsyncClient` with `ASGITransport` and manually initialize deps. No server startup required.

## Model Download Mirror

When downloading models for testing, use the mirror: `https://hf-mirror.com`
