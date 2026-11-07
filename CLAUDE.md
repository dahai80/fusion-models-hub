---
doc_id: fusion-models-hub/guide
product: fusion-models-hub
class: core
entry: service
doc_type: guide
title: { zh: "Claude Code 工作指引", en: "Claude Code guidance" }
audience: engineer
status: draft
version: 1.2.15
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - CLAUDE.md
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion-Model-Hub is the unified model repository and management center for the Fusion-MLX ecosystem on macOS Apple Silicon. It provides model discovery → download → conversion → quantization → version management → deployment → inference service across the Fusion ecosystem (Fusion-Studio, Fusion-Agent-Studio, Fusion-KB, Fusion-Bench).

**Critical constraint:** All inference, conversion, quantization, and verification is 100% delegated to the Fusion-MLX base via HTTP API. This project never imports `mlx`, `mlx-lm`, `torch`, or `transformers` directly. The `api/base_binding.py` module is the single verification point for Fusion-MLX availability.

## Development Commands

```bash
# Activate environment (always run first — shared monorepo venv)
cd /Users/dahai/fusion && source .venv/bin/activate

# Install in editable mode with test extras
pip install -e ".[test]"

# Run all tests (~1490 tests; pyproject addopts excludes the 4 integration files below)
pytest

# Run a single test file / single test
pytest tests/test_api.py -v
pytest tests/test_cache.py::test_cache_put_get -v

# Run with coverage
pytest --cov=fusion_model_hub --cov-report=term-missing

# Lint / format
ruff check .
ruff format .

# Integration tests (real PG/MinIO, real multi-node TCP, real MLX, Alembic-vs-ORM diff)
# Excluded from the default run via pyproject addopts — run explicitly with addopts cleared:
pip install -e ".[integration]"        # asyncpg / minio / psycopg / alembic
docker compose -f tests/integration/docker-compose-pg-minio.yml up -d
pytest tests/test_integration_pg_minio.py -o addopts=""

# Web console (web/ — Bun + Vite + React 18 + TypeScript + Tailwind)
cd web && bun install
bun run dev         # Vite dev server :11475
bun run build       # typecheck + vite build
bun run lint        # biome check src
bun run test        # bun test ./tests/ (own tests only; vendored tests run upstream)

# Start API server (subcommand `serve`, NOT bare invocation)
fusion-model-hub serve --host 127.0.0.1 --port 11444
# Or via lifecycle script (resolves shared .venv, writes .fusion-model-hub.pid)
./start.sh start|stop|restart|status

# CLI entry point (separate from server — discovery/download/recommend/analyze)
fmh list|download|recommend|analyze|hardware
```

Fusion-MLX lifecycle (for integration tests requiring a real model server):
```bash
~/claude-home/fusion-mlx/start.sh start   # Fusion-MLX server
~/claude-home/fusion-mlx/start.sh stop    # Stop
~/claude-home/fusion-mlx/start.sh status  # PID/port/memory/loaded models
```

## Architecture

The server is a FastAPI app factory (`server/app.py:create_app`) using `asynccontextmanager` lifespan. On startup it: initializes the async SQLAlchemy engine, runs `init_db`, wires dependencies via `init_deps`, enables auth, reconciles orphaned quantize tasks (RUNNING→FAILED, PENDING→requeued), and starts the backup scheduler.

```
fusion_model_hub/
├── db/                          # Async SQLAlchemy ORM layer (SQLite via aiosqlite, PG-ready)
│   ├── models.py                # 25+ tables: Model, ModelVersion, ModelTag, ModelRating, ModelFavorite,
│   │                            #   ModelBranch, QuantizeTask, LoraMergeTask, DistributedTask,
│   │                            #   Tenant, Role, ApiKey, AuditLog, ClusterNode, Webhook, Deployment,
│   │                            #   EvaluationResult, SecurityScan, Watermark, ApprovalRequest,
│   │                            #   GitLfsLock, DownloadTask
│   ├── database.py              # Async engine + session factory (get_engine / get_session_factory)
│   └── crud.py                  # Async CRUD with field whitelists (_MODEL_UPDATABLE etc.)
├── cache/                       # 3-level model cache (raw → converted → quantized)
│   ├── types.py                 # CacheEntry, CacheLevel, CacheStats
│   └── manager.py               # CacheManager: put/get/has/remove/gc/validate; index.json-backed,
│                                #   SHA256-verified, MLX-version-aware invalidation
├── server/                      # FastAPI server
│   ├── app.py                   # App factory: lifespan, CORS, request logging, global exception handler
│   ├── config.py                # Settings dataclass (env-driven, see Env Vars below)
│   ├── deps.py                  # DI: SessionDep, StoreDep, SettingsDep, CacheDep (+ init_deps/get_*)
│   ├── auth.py                  # API-key middleware: role/model/module ACLs, audit logging, rate limit
│   ├── identity.py              # #54 fusion-identity JWT integration (module-level enable flag, sync token verify)
│   ├── ssrf.py                  # validate_external_url — shared SSRF guard (sync router, downloads, downloader)
│   ├── events.py                # Webhook dispatch service layer (HMAC signing, retry, P1-16 bounded concurrency)
│   ├── eval_tasks.py            # Async eval runner (Fusion-Bench; R1 semaphore caps concurrent evals at 2)
│   ├── errors.py                # Shared error helpers
│   ├── rate_limit.py            # Per-key-prefix sliding-window QPS limiter (in-memory)
│   ├── metrics.py               # Prometheus counters/histograms + /metrics endpoint
│   ├── backup.py                # asyncio backup scheduler (auto-export models+versions to JSON)
│   ├── tasks.py                 # Async quantize runner (asyncio.create_task + _running_tasks dict)
│   ├── http_client.py           # H8 pooled httpx for MLX hot path (PoolClient + shared AsyncHTTPTransport)
│   ├── __main__.py              # CLI entry: serve / export / import / migrate subcommands
│   └── routers/                 # 30 routers, all mounted under /api/v1
│       ├── models.py            # Model CRUD + HF import + sync + batch + compare + publish + recommend
│       ├── versions.py          # Version lifecycle: chunked upload, URL download, benchmark, rollback, promote
│       ├── quantize.py          # Quantize task submit/monitor/list
│       ├── quantize_presets.py  # Preset quantization configs
│       ├── inference.py         # Inference proxy (chat/completions/embeddings) + loaded-model TTL eviction + FR-015 zero-downtime hot-reload (POST /models/{id}/hot-reload: preload new version, swap served record, dispatch model.hot_reloaded)
│       ├── cache.py             # 3-level cache management endpoints (/api/v1/cache/*)
│       ├── downloads.py         # Async model downloads from HF/ModelScope with progress + resume
│       ├── hardware.py recommend.py adapt.py analyze.py   # Hardware detection → recommendation → adaptation
│       ├── benchmarks.py        # Benchmark integration (Fusion-Bench)
│       ├── cluster.py           # Distributed node management + heartbeat
│       ├── tenants.py           # Multi-tenant isolation
│       ├── auth.py              # API key create/list/deactivate/delete
│       ├── webhooks.py          # Event webhooks CRUD; dispatch_webhook_event lives in server/events.py (router re-exports it)
│       ├── deployments.py       # Deployment tracking + MLX load/unload integration, gray release
│       ├── evaluations.py       # Evaluation task tracking
│       ├── approvals.py         # Model approval workflow (L1/L2/L3 levels)
│       ├── security.py watermark.py encryption.py   # Security scan, watermarking, encryption at rest
│       ├── ratings.py favorites.py branches.py     # Model ratings, favorites, version branches
│       ├── gitlfs.py            # Git-LFS lock management
│       ├── sync.py              # Registry sync (SSRF-protected source_url)
│       ├── monitor.py           # Runtime monitoring
│       └── system.py            # Health + storage stats + audit log
├── hardware/                    # HardwareDetector — queries Fusion-MLX for Apple Silicon profile
├── adapt/                       # Model adaptation: decision + migration between formats/quant levels
├── recommend/                   # Recommendation engine: scorer ranks models against hardware profile
├── storage/                     # StorageBackend ABC + LocalStore + MinioStore
├── sdk/                         # Python SDK: client.py (sync) + async_client.py + models.py
├── cli/                         # `fmh` CLI (typer): list, download, recommend, analyze, hardware
├── api/base_binding.py          # FusionMLXBase — detects/verifies Fusion-MLX availability via HTTP
├── convert/converter.py         # ModelConverter — HF/PyTorch/GGUF → MLX via Fusion-MLX
├── manage/manager.py            # LocalModelManager — register/list/activate/delete local models
├── utils/                       # Shared helpers
└── repo/                        # In-memory catalog + async downloader + ModelScope search
    ├── registry.py models.py downloader.py modelscope_search.py
```

### Key Design Decisions

- **FastAPI lifespan (not on_event):** App uses `asynccontextmanager` lifespan. Tests manually call `init_deps()` since `httpx` ASGITransport doesn't trigger ASGI lifespan.
- **Pooled httpx for the MLX hot path (H8, `server/http_client.py`):** The inference router AND the cluster router do `from .. import http_client as httpx` and read `httpx.AsyncClient` (== `PoolClient`, an `httpx.AsyncClient` subclass). One `AsyncHTTPTransport` per base_url is kept alive process-wide; each `async with httpx.AsyncClient(timeout=X)` builds a thin wrapper whose `aclose` is a no-op, so connections are reused across serve/chat/unload instead of torn down per call. Per-call `timeout` passes through to the wrapper, so deadlines stay independent. Low-frequency / external-host callers (sync registry, HF search, downloads, convert, webhooks, SDK) keep using real `httpx` directly — only the MLX hot path opts into pooling. `close_all_transports()` runs in the lifespan shutdown. Test patches `patch("...inference.httpx.AsyncClient", return_value=mock)` still intercept because the mock supplies its own `__aenter__`/`__aexit__`/`post`; no test rewrites needed. **cluster.py coverage was a gap found by the multi-node scale test** (`tests/integration/multinode/`) — migrating it from raw `httpx` to the pooled client gave 5.6× throughput / 6× lower p50 on the cluster-routing path (S1: 16.69→93.45 rps, 881→148ms).
- **Cluster routing is round-robin load-balance with per-call failover (`route_inference`):** `mode=cluster` skips local MLX, builds the active node list (`list_cluster_nodes(session)` = `order_by(ClusterNode.created_at.desc())`, filtered by `_effective_status == "active"`), then rotates the start offset per call via a monotonic module counter `_round_robin_counter` (`offset = next(counter) % len(active); active = active[offset:] + active[:offset]`) so load spreads evenly across healthy nodes. Each call still iterates the whole rotated list and `return`s on the first node whose `_chat` succeeds — failover is preserved within a single call if the round-robin start node is down. Single asyncio event loop → no lock; `itertools.count` is deterministic (no `random`). `mode=auto` uses local MLX if `_check_alive` else falls to cluster; `mode=local` forces local. (Fix for issue #31: before, the loop started at `nodes[0]` every call, so the newest-registered node absorbed all traffic and earlier primaries served 0 — the scale test measured real-mlx at 0/150.)
- **Dependency injection:** `SessionDep`, `StoreDep`, `SettingsDep`, `CacheDep` are `Annotated` types via FastAPI `Depends`. Module-level singletons in `deps.py` initialized by `init_deps(settings, engine)`. Routers use `get_*()` accessors, never touch the private globals.
- **3-level cache (`cache/`):** raw → converted → quantized, keyed `model_id:level[:Nbit]`. `index.json` persists entry metadata (path/sha256/size/mlx_version/last_accessed/ref_count). `CacheManager.validate()` invalidates converted+quantized entries whose `mlx_version` differs from current — version-aware. Wired into `_run_quantize()` in `tasks.py` as check-then-put; the `adapt` router's convert path is intentionally uncached (surgical separation).
- **Async task management:** Quantize tasks use `asyncio.create_task` with `_running_tasks` dict tracking + `add_done_callback` cleanup. On startup, `_reconcile_orphaned_tasks()` fails RUNNING tasks orphaned by restart and requeues PENDING ones. Tasks update DB status on completion/failure.
- **LoRA adapters (#22):** `Model.base_model_id` (FK → `models.id`, nullable) links a LoRA adapter model to its base LLM; `model_type=LORA` + non-empty `base_model_id` required at create (validated, 404 if base missing). `_MODEL_UPDATABLE` whitelists `base_model_id` so it can be repointed via `PUT /models/{id}`. `_run_merge` (`routers/quantize.py`) calls `POST {mlx_url}/v1/models/{name}/merge-adapter` to fuse adapter → new `ModelVersion` (sets `output_version_id`); 404-tolerant until fusion-mlx ships the endpoint (issue fusion-mlx#584) — surfaces a clear "upgrade fusion-mlx" FAILED. The column ships in the squashed initial migration `0f2330f0ac47`; `init_db` create_all covers the test path.
- **Auth middleware:** `auth_enabled` defaults **True**. Write methods (POST/PUT/DELETE/PATCH) require `X-API-Key`. `PUBLIC_PATHS` exempt `/api/v1/system/health`, `/docs`, `/openapi.json`, `/api/v1/auth/keys` (bootstrap). RBAC via `UserRole`: admin=all, developer=no-DELETE, viewer=read-only. Per-key ACLs on `allowed_models` (path-extracted model_id) and `allowed_modules` (`X-Fusion-Module` header). Audit log written for all non-inference writes.
- **Multi-tenant isolation:** `tenant_id` on Model, ApiKey, AuditLog, Webhook, Deployment, EvaluationResult, ApprovalRequest. API keys carry tenant context into `request.state`.
- **Storage abstraction:** `StorageBackend` ABC with `LocalStore` (chunked 5MB upload + SHA256 assemble) and `MinioStore` (S3-compatible). Selected by `storage_type` setting in `init_deps`.
- **Version lifecycle:** draft → testing → published → deprecated → retired state machine. New models default to `ModelStatus.DRAFT` and must be `publish`ed before inference serves them.
- **Field whitelists:** `update_model`/`update_version`/`update_quantize_task` use explicit allowlist sets (`_MODEL_UPDATABLE`, `_VERSION_UPDATABLE`, `_TASK_UPDATABLE`) instead of `hasattr` — prevents accidental field overwrites.
- **SSRF prevention:** `sync_registry` source_url and `UrlDownloadRequest.url` validate scheme (http/https only) and block internal IPs (localhost, 10.x, 172.16-31.x, 192.168.x, 169.254.169.254).
- **Download resume:** `ModelDownloader` preserves `.part` files on failure and sends HTTP Range headers to resume from last byte on retry.
- **Rate limiting:** In-memory sliding-window (60s) per `ApiKey.key_prefix`, enforced by `qps_limit` (0 = unlimited). Not persisted across restart.
- **Webhooks:** HMAC-SHA256 signed delivery with retry + exponential backoff. Events dispatched via `dispatch_webhook_event(event, payload)` from `routers/webhooks.py` — called by quantize completion, version publish, precision-loss warnings, FR-015 hot-reload (`model.hot_reloaded`), #22 LoRA adapter publish/merge (`adapter.published` on LoRA model create, `adapter.merged` on `_run_merge` completion). `WebhookEvent` enum (`db/models.py`) lists all event slugs; `Webhook.events` is comma-separated, matched by substring.
- **Auto-backup:** `backup.py` scheduler runs in an asyncio loop at `backup_interval_seconds`; disabled if `backup_dir` unset. Exports models + versions to JSON.
- **Observability:** Prometheus metrics (`fmh_http_requests_total` etc.) via `metrics_middleware` and `/metrics` endpoint.
- **SQLite → PostgreSQL migration path:** All DB access through async SQLAlchemy ORM; `db_url` swap is the only change. Alembic migrations live in `alembic/` (sync engine, `FMH_ALEMBIC_URL` env); run via `fusion-model-hub migrate` or `alembic upgrade head`. **ORM is the single source of truth:** `alembic/versions/` holds ONE squashed initial migration (`0f2330f0ac47`, autogenerated from `Base.metadata`) — do NOT hand-write `op.add_column` migrations that drift from the ORM; regenerate via `alembic revision --autogenerate` instead. The `tests/test_integration_migration.py` gate (excluded from the default run, see Test Structure) enforces this: it runs `alembic upgrade head` on a fresh PG and diffs the result against `Base.metadata.create_all`, failing on any divergence. Run it explicitly with `docker compose -f tests/integration/docker-compose-pg-minio.yml up -d` then `pytest tests/test_integration_migration.py -o addopts=""`.

## Environment Variables

All optional, prefixed `FMH_` (auth uses `MODEL_HUB_AUTH_ENABLED`):

| Var | Default | Purpose |
|-----|---------|---------|
| `FMH_DATA_DIR` | `./data` | Data root (DB, storage, cache) |
| `FMH_CACHE_DIR` | `<data_dir>/cache` | 3-level cache root |
| `FMH_MLX_URL` | `http://127.0.0.1:11432` | LLM API URL — fusion-gateway by default (#58); chat/completions/embeddings/health route through it |
| `FMH_MLX_TOOL_URL` | `http://127.0.0.1:11434` | Direct fusion-mlx URL for tool endpoints (hardware/recommend/analyze/adapt/convert/quantize/benchmarks) and model load/unload — the gateway does not route these |
| `FMH_MLX_GATEWAY_API_KEY` | empty | Key the gateway accepts for Hub LLM calls. Empty = reuse `mlx_internal_api_key` (single-key installs) |
| `MODEL_HUB_AUTH_ENABLED` | `true` | Enable API-key auth |
| `FMH_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `FMH_MAX_UPLOAD_SIZE_MB` | `50000` | Max upload (50GB) |
| `FMH_STORAGE_TYPE` | `local` | `local` or `minio` |
| `FMH_MINIO_ENDPOINT` / `FMH_MINIO_ACCESS_KEY` / `FMH_MINIO_SECRET_KEY` / `FMH_MINIO_BUCKET` / `FMH_MINIO_SECURE` | empty | MinIO storage config |
| `FMH_BACKUP_DIR` / `FMH_BACKUP_INTERVAL` | empty / `86400` | Auto-backup |
| `FMH_TLS_CERTFILE` / `FMH_TLS_KEYFILE` | empty | TLS |
| `FMH_BENCH_URL` / `FMH_BENCH_AUTO_TRIGGER` | `http://localhost:8090` / `false` | Fusion-Bench integration |
| `FMH_BENCH_API_KEY` | empty | X-API-Key for Hub→Fusion-Bench task creation (anonymous VIEWER lacks TASK_CREATE → 403; set an operator/admin bench key) |
| `FMH_EVAL_RUNNER_ENABLED` | `true` | Gate the async eval runner. `false` = POST /evaluations creates a PENDING row but does not spawn the Fusion-Bench runner |
| `FMH_IDENTITY_INTEGRATION_ENABLED` | `false` | #54: authenticate inbound requests via fusion-identity JWT (sole JWT issuer + tenant registry) instead of local API keys |
| `FMH_IDENTITY_URL` / `FMH_IDENTITY_SERVICE_TOKEN` / `FMH_IDENTITY_JWT_ISSUER` / `FMH_IDENTITY_JWT_AUDIENCE` | `http://127.0.0.1:11470` / empty / `fusion-identity` / `fusion-cluster` | fusion-identity connection config |
| `FMH_DOWNLOAD_SPEED_LIMIT` | `0` | Download speed limit (kbps, 0=none) |
| `FMH_PRECISION_LOSS_THRESHOLD` | `10.0` | Quantize precision-loss alert % |
| `FUSION_MLX_API_KEY` | empty | Bearer token for Hub→MLX requests. #68: resolution delegated to `fusion_core.config` — env `FUSION_MLX_API_KEY` > `~/.fusion-mlx/settings.json` `auth.api_key` > `FUSION_CLIENT_KEY`/`MLX_API_KEY` chain (dev warns, prod raises). Deprecated bare `MLX_INTERNAL_API_KEY` alias REMOVED |
| `FMH_ALEMBIC_URL` | `sqlite://` | Sync DB URL for Alembic migrations |

**MLX URL note (dual-URL, #58):** `mlx_url` (LLM surface: chat/completions/embeddings/health) defaults to `http://127.0.0.1:11432` — fusion-gateway, which owns routing policy, rate limits and audit; Hub presents `FMH_MLX_GATEWAY_API_KEY` (falls back to the MLX internal key). `mlx_tool_url` (tool surface: hardware/recommend/analyze/adapt/convert/quantize/benchmarks, cluster peers, deployment metrics, and model load/unload) defaults to `http://127.0.0.1:11434` — direct fusion-mlx with the MLX internal key. The tool surface stays direct because the gateway does not route those endpoints (upstream fusion-gateway issue), and its `/v1/models/{id}/load|unload` intercept rewrites into the Hub's own `POST /api/v1/models/{id}/serve` — sending Hub load/unload through the gateway would 404 on unknown IDs and loop forever on known ones. History: `d0aa3f8` set everything to `11432` and broke (gateway rejected Hub API keys); the #58 fix is the explicit split above, not a silent default. Override either URL via env; point both at the gateway only once it routes the tool surface.

## Dependencies

- `fastapi>=0.115.0` — async REST framework
- `uvicorn[standard]>=0.30.0` — ASGI server
- `sqlalchemy[asyncio]>=2.0.0` — async ORM
- `aiosqlite>=0.20.0` — async SQLite driver
- `python-multipart>=0.0.9` — form data parsing
- `httpx>=0.27.0` — async HTTP client (all Fusion-MLX + SDK)
- `pydantic>=2.0.0` — data validation
- `prometheus_client>=0.20.0` — metrics
- `cryptography>=42.0.0` — encryption/watermarking
- `typer>=0.12.0` — CLI (`fmh`)
- `setproctitle>=1.3.0,<2.0.0` — process title (`fusion-model-hub serve :<port>` in ps; try/except fallback in `server/__main__.py`)
- Python >=3.12

## Test Structure

~1490 tests across 25+ files. Tests use `httpx.AsyncClient` with `ASGITransport` and manually initialize deps — no server startup required. Four integration files are excluded from the default run via `pyproject.toml` `addopts` (`--ignore`): `test_integration_pg_minio.py`, `test_integration_multinode.py`, `test_integration_mlx.py`, `test_integration_migration.py` — run each explicitly by path with `-o addopts=""` (backing services or bound ports required; `pip install -e ".[integration]"` adds the drivers).

- `tests/test_api.py` — core API integration (all routers, in-memory SQLite)
- `tests/test_cache.py` — 3-level cache manager
- `tests/test_core.py` — repo/convert/manage/api unit tests
- `tests/test_sdk.py` / `tests/test_sdk_async.py` / `tests/test_sdk_models.py` — Python SDK
- `tests/test_routers_deep.py` / `tests/test_routers_extended.py` — router deep coverage (incl. FR-015 hot-reload `TestInferenceHotReload`, #22 LoRA `TestLoraAdapter` — base_model_id CRUD, adapter.published/adapter.merged webhooks, _run_merge 404→FAILED + 200→COMPLETED+version)
- `tests/test_enterprise.py` — multi-tenant/RBAC/webhooks/deployments
- `tests/test_audit_p0_fixes.py` — audit P0 fixes (security/audit-log regressions)
- `tests/test_cov_*.py` — per-area coverage suites (db_core, inference_cluster, quantize_tasks, routers_misc, sdk_cli, versions_app)
- `tests/test_eval_runner.py` — async eval runner (semaphore bounds, FMH_EVAL_RUNNER_ENABLED gate)
- `tests/test_identity_integration.py` — #54 fusion-identity JWT auth path
- `tests/test_new_features.py` / `test_coverage*.py` — feature + coverage boosters
- `tests/test_studio_compat.py` — fusion-studio DTO contract (envelopes, flat hardware, health fields, deployment camelCase + scale/name aliases, deployment metrics keys + live-fields-from-/v1/metrics/json, benchmark template->suite alias)
- `tests/test_config.py` — Settings.mlx_internal_api_key resolution order via fusion_core.config (#68: FUSION_MLX_API_KEY > settings.json auth.api_key > FUSION_CLIENT_KEY/MLX_API_KEY chain; dev warn / prod raise; bare MLX_INTERNAL_API_KEY ignored)

**Auth-enabled test gotcha:** With `auth_enabled=True` (default), DRAFT models are unservable and write endpoints require API keys. Test fixtures commonly disable auth via `set_auth_enabled(False)` and publish models post-creation. `/api/v1/auth/keys` is in `PUBLIC_PATHS` for bootstrap key creation.

## Web Console (web/)

React 18 + TypeScript + Vite + Tailwind, built and tested with Bun (see commands above). Four pages over the REST API on `127.0.0.1:11444` — Models (search/detail/HF import), Downloads (progress/speed/ETA, cancel), Quantize (submit + task table), Storage (usage tiles + MLX health). Write endpoints need `X-API-Key` (same auth as the REST API). Vendored `@fusion/web-sdk` + `@fusion/web-ui` live under `web/vendor/` as `file:` deps, re-synced from fusion-web-shell releases — do NOT edit them in place; re-sync from upstream instead. Lint is Biome (`bun run lint`), not ruff.

## Repo Conventions (from AGENTS.md)

- Port discipline: `architecture/port-registry.yaml` (monorepo root) is the single source of truth — register before binding, never reuse a claimed port. This service: 11444 (API), 11475 (web dev server).
- Process naming: long-running processes must set their title to the product name (`fusion-model-hub serve :11444`), not leave a bare `python`/`node` in `ps`. Use `setproctitle` (with try/except fallback) on macOS homebrew Python — `exec -a` is unreliable there.
- Cross-repo changes: implement only in this repo; for upstream problems (fusion-mlx, fusion-web-shell, fusion-gateway), file an issue on that repo documenting source, contract, and blocking relationship, then comment the issue number on the parent issue.

## Model Download Mirror

When downloading models for testing, use the mirror: `https://hf-mirror.com`
