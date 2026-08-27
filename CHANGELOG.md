# Changelog

All notable changes to **fusion-model-hub** are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions follow [PEP 440](https://peps.python.org/pep-0440/).

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

1. Watermark embed into model weights (currently DB row + signature only).
2. Sync push/pull real file bytes (currently metadata only).
3. ~~Evaluations async execution runner (currently DB row, stuck PENDING).~~ **Done** (PR #46, `3f16e4c`): `server/eval_tasks.py` async runner (`submit_evaluation` → `asyncio.create_task`, `Semaphore(2)` bound) POSTs a Fusion-Bench task, polls to terminal, fetches result `metric_value` → `score`, flips PENDING→RUNNING→COMPLETED/FAILED. `bench_api_key` (`FMH_BENCH_API_KEY`) + `eval_runner_enabled` (`FMH_EVAL_RUNNER_ENABLED`) config; bench auth wired into benchmark trigger + quantize auto-trigger. Startup reconciliation fails orphaned RUNNING evals + resumes PENDING. 7 new tests, 1453 total.
4. ~~System `scan_duplicates`/`disk_cleanup` real delete (currently identify only).~~ **Done** (PR #47, `5e80da6`): both endpoints now take `dry_run` (default `true` = safe identify, prior behavior). `dry_run=false` reclaims disk via the storage backend — `scan-duplicates` retires redundant duplicate-weight versions (keeps oldest), `cleanup` deletes retired versions' files; DB rows kept for provenance, `file_path`/`file_size` cleared. 4 new real-delete tests, 1457 total.
5. ~~SDK missing router method groups + `AsyncFusionHubClient` not in `__init__.__all__`.~~ **Done** (PR #45, `0a282e7`): ~40 methods added per sync+async client (serve lifecycle, cache, deployments, downloads, evaluations, tenants/roles, webhooks, monitor); `_patch` helper + `_delete` params; `AsyncFusionModelHubClient` exported. 1446 tests.
6. Per-inference ~4 DB roundtrips reduction.

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
