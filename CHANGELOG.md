# Changelog

All notable changes to **fusion-model-hub** are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions follow [PEP 440](https://peps.python.org/pep-0440/).

## [1.1.1] — 2026-09-02

Patch release: Dockerfile build fix + gateway-origin tenant-scoping enforcement.

### Fixes

- **#52 Dockerfile build fails on `python:3.12-slim`** — the Dockerfile installed the full monorepo `requirements.lock` (~200 pins spanning every fusion-* sub-project), including sdist-only packages (`miniaudio`, `mflux`) that need a C++ compiler absent on the slim base, so `docker build` died on `FileNotFoundError: 'c++'`. Replaced the monorepo lock install with `pip install .` (resolves model-hub's own 10 runtime deps, all prebuilt wheels, no compiler needed). model-hub is not an MLX node, so the "every node identical" lock rationale does not apply; a per-service dep set is more correct and yields a smaller image. The Dockerfile build context is now the sub-project dir (no monorepo root needed). Verified: image builds clean, `/api/v1/system/health` returns 200 in a container. (`Dockerfile`)
- **#53 enforce gateway-origin + `X-Fusion-Tenant` scoping** — backend-side half of fusion-gateway #150 (Gap 1c, multi-tenant isolation). Added `gateway_origin_enforced` setting (env `FMH_GATEWAY_ORIGIN_ENFORCED`, default OFF for single-tenant dev). When ON, a `/api/v1/*` request MUST carry `X-Fusion-Route: gateway-decision` (the header the gateway stamps on every outbound request) or it is rejected 403, blocking direct-port bypass of the gateway's tenant derivation. `X-Fusion-Tenant`, when present, overrides the api_key's `tenant_id` as the authoritative tenant for the request (gateway derives it from the key→team binding); `X-Space-Id` is treated as non-authoritative passthrough (never read). Health/docs stay public so liveness probes and the OpenAPI UI keep working behind a gateway. (`server/auth.py`, `server/config.py`, `server/app.py`)

## [1.1.0] — 2026-09-01

**General availability** release. Promotes `1.1.0rc1` (enterprise release-ready verdict) to GA with one follow-up fix. 1470 tests / 91% cov, P0=0 P1=0, ruff clean.

### Fixes

- **#51 exact-name lookup on `GET /api/v1/models`** — added `name=` query param for exact, case-sensitive match via the same `crud.get_model_by_name` the POST 409 collision check uses, tenant-scoped. A publisher that gets a 409 on `POST /models` could not previously tell an idempotent re-publish (same model) from a fuzzy `keyword=` superset sharing a token. `GET /api/v1/models?name=X` now returns at most one exact match (empty `items` if none), enabling idempotent publish detection. (`server/routers/models.py`)
- **flaky `test_compare_no_output_version` under full-suite load** — the un-mocked quantize runner raced the module-level `_QUANTIZE_CONCURRENCY` semaphore and a real MLX HTTP timeout under full-suite load, so a fixed 0.3s sleep did not always reach FAILED. Mocked `ModelConverter.quantize` for a deterministic fast-fail and replaced the sleep with a poll loop. (`tests/test_cov_quantize_tasks.py`)

## [1.1.0rc1] — 2026-08-27

First **release candidate** after the 6-dimension enterprise audit. Baseline `1.0.18` (commit `21310f6`, 890 tests / 81% cov, 2 P0 + 22 P1) → RC `1.1.0rc1` (commit `c5dbbde`, 1410 tests / 91% cov, P0=0 P1=0). Re-audit verdict: **enterprise release-ready** (report at `audit/fusion-model-hub-reaudit-0827.md`).

### Audit Remediation — P0 (2)

- **P0-1 auth PBKDF2 blocking event loop** — raw key → cached key_hash, uncached branch offloaded via `anyio.to_thread.run_sync`; `last_used` write throttled so verify stays read-only. (`db/crud.py`, `server/auth.py`)
- **P0-2 hot-path queries zero index** — 20+ `Index(...)` on `models.tenant_id`, `model_versions.model_id`, `quantize_tasks.status`, `api_keys.key_hash`, `audit_logs.created_at`, `cluster_nodes.status`; migration `426bf65ff049`. (`db/models.py`)

### Audit Remediation — P1 (22)

- **Functional (5):** quantize_presets no longer stuck PENDING; gitlfs batch hrefs reachable (PUT/GET/verify routes); role CRUD admin-guarded; encryption streamed off-loop (64MB chunks); quantize provenance `output_hash`/`output_size` recorded with local fallback.
- **Async IO offload (5):** cache `put/gc/validate`, local_store chunk write/assemble, `hash_file`, version upload/export/import, encryption — all via `to_thread.run_sync`, no loop blocking.
- **Security (3):** upload filename path-traversal double sanitize (basename + reject `.`/`..`); cross-tenant API-key forgery blocked (tenanted admin 403 on tenant mismatch); TLS config wired (`FMH_TLS_CERTFILE`/`FMH_TLS_KEYFILE` → uvicorn ssl).
- **Performance (2):** backup N+1 eliminated (single query, group by model); MLX hot path pooled httpx (`PoolClient` + process-wide `AsyncHTTPTransport`, per-call timeout passthrough, no-op `aclose`).
- **Fault tolerance (4):** webhook fire-and-forget (`asyncio.create_task` + strong-ref); lora-merge atomic (task + new version together); chunk upload per-chunk + total cap; `init_db` failure degrades with log, no process crash.
- **Ops (3):** Dockerfile `CMD ["fusion-model-hub","serve"]` (subcommand, not bare); env→Settings full mapping + start.sh passthrough; backup `import` subcommand restores models+versions from JSON.

### Audit Remediation — P2/P3 (9 of 15, post-release backlog for the rest)

- Round-robin cluster routing with per-call failover (#31); downloads cooperative cancel (#29); H8 pooled httpx multi-node scale.
- Module ACL fail-closed (#9 — restricted key + no `X-Fusion-Module` → 403); `_engines` registry `threading.Lock` (#10); `start.sh` graceful drain (`FMH_DRAIN_TIMEOUT`, SIGKILL only on timeout, #13); L3 approval quorum (`approvers` column, `APPROVAL_L3_QUORUM=2` distinct approvers, migration `24413e374e1c`, #7); branch merge→version (#5).
- Quantize MLX contract alignment (#40, fusion-mlx#646 consumer-side): `model`+`output_path`+`quant_bits` to `/v1/quantize`; async job poll `/v1/quantize/jobs/{id}` until `done`; layered routes remapped to real endpoints; converter + layered routes carry MLX Bearer.

### Regression Fixes (Re-audit, PR #42)

- **quantize false-complete** — sync-response coercion narrowed to missing-status-only; an explicit `failed`/`error` with a partial `output_path` no longer flips to `completed` (would create a corrupt `ModelVersion`). (`convert/converter.py`)
- **layered routes missing MLX auth** — `start_layered_quantize`/`get_layered_quantize_job`/`list_layered_quantize_jobs` now carry `_mlx_headers(settings)` Bearer; previously 401 while converter path succeeded. (`server/routers/quantize.py`)

### Tooling

- `ruff format` baseline applied (94 files, formatting-only); `ruff check` clean. (PR #43)

### Verification (this RC)

- **Tests:** 1410 passed, 0 failed.
- **Coverage:** 91% (754 missing / 8855 total).
- **Lint:** `ruff check` 0 errors; `ruff format --check` clean.
- **Alembic ORM consistency:** 0 drift (fresh `upgrade head` vs `create_all`); migration chain `0f2330f0ac47` → `7e8a9f01b2c4` → `426bf65ff049` → `24413e374e1c` (head).
- **Real fusion-mlx integration smoke (2026-08-27):** health (`mlxConnected:true`) + inference (Llama-3.2-1B chat, real model reply) + quantize (4bit→8bit async job `completed`, 712MB output + provenance) + Bearer auth (`dahai168`, no-key→401). All Hub→MLX hot-path calls carry Bearer.

### Post-release Backlog (6 P2/P3, not in this RC)

1. ~~Watermark embed into model weights (currently DB row + signature only).~~ **Done** (PR #49, `8313aef`): watermark now also writes a signed `watermark.json` sidecar into the version dir so it travels with the model files (survives copy/sync, verifiable without the Hub DB). `StorageBackend` gains `write_sidecar`/`read_sidecar` (LocalStore atomic write; MinioStore `NotImplementedError` → 501). `verify` is defense-in-depth: when both sidecar + DB exist, **both** must verify (tamper in either → `verified:False`); single-source paths (copied model / legacy) decide alone. Constant-time `hmac.compare_digest` preserved. Tensor-level weight embed tracked upstream in fusion-mlx#656 (filed). 4 new tests, 1463 total.
2. ~~Sync push/pull real file bytes (currently metadata only).~~ **Done** (PR #50, `eff0c4e`): sync now streams the real weight file bytes, not just model metadata. New `POST /sync/receive` endpoint is the symmetric remote-side target for push — idempotent (get-or-create model by id/name + version, stream uploaded file into the store, set `file_path`/`file_hash`/`file_size`; re-push overwrites, no 409). `POST /sync/push` streams the local version file to `{target}/api/v1/sync/receive` as multipart after the metadata import; per-version status now distinguishes `pushed`/`metadata_only`/`partial`/`failed`. `POST /sync/pull` streams `GET {source}/api/v1/versions/{remote_vid}/download` into the local store (1MB chunks, inline SHA256, `max_upload_size_mb` enforced, partial-file cleanup on over-limit/crash) + sets the version file fields; get-or-create model (re-pull reuses, no `already_exists`). `_find_version_by_label` queries the row directly (no lazy `m.versions` load → `MissingGreenlet` on a rolled-back session); model id captured as a string before `create_version`. Metadata-only versions (no `file_path`) still sync as `metadata_only` — backward compatible. 5 new tests + 3 existing updated, 1468 total.
3. ~~Evaluations async execution runner (currently DB row, stuck PENDING).~~ **Done** (PR #46, `3f16e4c`): `server/eval_tasks.py` async runner (`submit_evaluation` → `asyncio.create_task`, `Semaphore(2)` bound) POSTs a Fusion-Bench task, polls to terminal, fetches result `metric_value` → `score`, flips PENDING→RUNNING→COMPLETED/FAILED. `bench_api_key` (`FMH_BENCH_API_KEY`) + `eval_runner_enabled` (`FMH_EVAL_RUNNER_ENABLED`) config; bench auth wired into benchmark trigger + quantize auto-trigger. Startup reconciliation fails orphaned RUNNING evals + resumes PENDING. 7 new tests, 1453 total.
4. ~~System `scan_duplicates`/`disk_cleanup` real delete (currently identify only).~~ **Done** (PR #47, `5e80da6`): both endpoints now take `dry_run` (default `true` = safe identify, prior behavior). `dry_run=false` reclaims disk via the storage backend — `scan-duplicates` retires redundant duplicate-weight versions (keeps oldest), `cleanup` deletes retired versions' files; DB rows kept for provenance, `file_path`/`file_size` cleared. 4 new real-delete tests, 1457 total.
5. ~~SDK missing router method groups + `AsyncFusionHubClient` not in `__init__.__all__`.~~ **Done** (PR #45, `0a282e7`): ~40 methods added per sync+async client (serve lifecycle, cache, deployments, downloads, evaluations, tenants/roles, webhooks, monitor); `_patch` helper + `_delete` params; `AsyncFusionModelHubClient` exported. 1446 tests.
6. ~~Per-inference ~4 DB roundtrips reduction.~~ **Done** (PR #48, `8563c61`): hot path (`chat`/`completions`/`embeddings`) dropped from ~4 DB sessions/call to ~1. `_check_module_access` + `_resolve_model_name_for_inference` now reuse the caller's fetched model + request session (legacy no-args fallback preserved); per-call audit insert deferred to a fire-and-forget `asyncio.create_task` (strong-ref `_pending_audit_tasks` set, RUF006-safe) off the critical path. 2 new tests (audit-deferred, no-extra-session), 1459 total.

---

## [1.0.18] — 2026-08-26

- `2afcb40` dispose async engines (silence aiosqlite runtime warning).
- `ad95f84` v1.0.17 production-readiness blockers + minor items (PR #35).

## [1.0.16] — 2026-08-25

- `193ae24` release: H8 pooled httpx (#71), download cancel (#29), round-robin routing (#31).

## [1.0.14] / [1.0.13] / [1.0.12]

- Incremental feature + fix releases.

## [1.0.2] / [1.0.1]

- Early access builds.

## [0.2.0]

- Initial pre-1.0 preview.
