# Changelog

All notable changes to **fusion-model-hub** are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions follow [PEP 440](https://peps.python.org/pep-0440/).

## [1.2.7] — 2026-10-28

Patch release: health-check recursion fix (#71).

### Fixed
- **#71**: `/api/v1/system/health` probed the gateway's full `/health`, which fans out to every backend including the hub itself — an unbounded probe cycle (gateway → hub → gateway → …) truncated only by stacked 5s timeouts (13-20s observed). The MLX subcheck now probes the gateway's lightweight `/healthz` (no backend fan-out) first, falling back to `/health` only on 404 (direct fusion-mlx targets have no `/healthz`). Non-404 `/healthz` failures do not trigger the fallback, so an erroring gateway cannot re-open the recursion. Health now returns in milliseconds with the gateway running.

## [1.2.6] — 2026-10-28

Patch release: config governance (#68).

### Changed
- **#68**: `mlx_internal_api_key` resolution delegated to `fusion_core.config` (fusion-core#31) — env `FUSION_MLX_API_KEY` > `~/.fusion-mlx/settings.json` `auth.api_key` > `FUSION_CLIENT_KEY`/`MLX_API_KEY` chain. Under `FUSION_PROFILE=prod` an unresolvable key now fails fast at startup (config-conventions §3) instead of silently sending no Bearer; dev keeps the warn-and-continue behavior.
- **#68 (BREAKING for stale deployments)**: the deprecated bare `MLX_INTERNAL_API_KEY` env alias is removed — it is no longer read. Migrate to `FUSION_MLX_API_KEY` (or rely on the settings.json fallback) before upgrading.
- `fusion-core` added as an explicit dependency (git-pinned `v0.7.0`, fusion-bench pattern); the E-S13 lax-perms warning on `~/.fusion-mlx/settings.json` stays local because `fusion_core.config` does not check file permissions.

## [1.2.5] — 2026-10-28

Patch release: security remediation + operability.

### Fixed

- **#63 Hardcoded default secrets removed from encryption/watermark routers** — `FMH_ENCRYPTION_KEY` / `FMH_WATERMARK_SECRET` no longer compare against source-public default literals (flagged by `scripts/check-env.py`, audit §1.2). New shared helper `resolve_service_secret()` in `server/config.py` implements config-conventions §3 profile semantics: `FUSION_PROFILE=prod` with a missing secret raises `RuntimeError` with a three-element guidance message (variable name + how to obtain + doc path); dev (default) allows empty with a prominent `logger.warning` and the endpoints keep their 503 refusal. `scripts/check-env.py` no longer reports this repo in non-baseline secret errors.
- **#62 Process title set to `fusion-model-hub serve :<port>`** — the serve process previously showed as bare homebrew `Python` in `ps`/Activity Monitor. Now titled via `setproctitle` (try/except fallback; `setproctitle>=1.3.0,<2.0.0` added to core dependencies), matching the fusion-gupiao / fusion-k12-teacher pattern.
- **Test flake: bootstrap rate limiter state leaked across tests** — the module-global in-memory rate limiter (60s window) tripped `429 Bootstrap rate limit exceeded` for tests hitting `/api/v1/auth/keys` later in the same window under `pytest-randomly` (~2/3 of full-suite runs had 1-2 failures, each green in isolation). Autouse conftest fixture now calls `reset_rate_limits()` around every test; full suite green 3/3 consecutive runs.

## [1.2.4] — 2026-10-27

Patch release: re-sync vendored `@fusion/web-sdk` to upstream v0.1.2 (#60).

### Fixed

- **#60 Vendored `@fusion/web-sdk` re-synced to v0.1.2** — picked up two upstream fixes while keeping the local #1298 (`?embedded=1` deep-link posture) and #20 (direct-request content-type/timeout) enhancements:
  - Host bridge TypeError (fusion-web-shell#75, v0.1.1): `'__httpError' in (result as object)` threw `TypeError: Cannot use 'in' operator` when a bridge method returned a bare string (e.g. `theme` → `'light'`), so the module never received its response. Now guarded by the `isHttpErrorResult()` type predicate.
  - Stale `theme.get()` in embedded mode (fusion-web-shell#77, v0.1.2): `theme.get()` read the module's own `documentElement`, which never carries the host theme class. Now a synchronous cached read (`createThemeState` in new `src/theme.ts`), seeded from the module document, corrected by a background host pull at init, kept fresh by host push events; same-value updates don't re-notify. API unchanged: `get(): ThemeMode` stays sync, `subscribe` is the source of truth. Local extension: the background host pull is skipped when there is no parent frame (#1298 deep-link posture) — no host can answer, so the pull would just dangle on its 5s timeout.
  - Vendored `package.json` bumped 0.1.0 → 0.1.2; upstream `tests/host.test.ts` + `tests/theme.test.ts` vendored alongside the existing suite (web tests 34 → 42, all pass; `tsc --noEmit` + `bun run build` clean). No wire-format or API changes; the module app had no hand-rolled theme pulls to migrate.

## [1.2.3] — 2026-10-26

Patch release: route Hub LLM calls through fusion-gateway (:11432) instead of direct fusion-mlx (:11434) defaults (#58, tracked from fusion-gateway#301 AC3).

### Changed

- **#58 Dual-URL MLX routing — LLM traffic defaults to fusion-gateway** — `mlx_url` now defaults to `http://127.0.0.1:11432` (the gateway owns routing policy, overflow ceilings, audit and rate limits, so Hub chat/completions/embeddings/health must not silently bypass it), with a new `FMH_MLX_GATEWAY_API_KEY` setting (env `FMH_MLX_GATEWAY_API_KEY`, falls back to `mlx_internal_api_key` for single-key installs) presented as the Bearer on LLM-surface calls — the gateway's key store rejects the raw MLX admin key. New `mlx_tool_url` (env `FMH_MLX_TOOL_URL`, CLI `--mlx-tool-url`) keeps the non-LLM tool surface (hardware/recommend/analyze/adapt/convert/quantize/benchmarks, `api/base_binding.py`) on direct fusion-mlx `:11434`, explicit rather than silent, because the gateway does not route those endpoints (upstream fusion-gateway issue filed for passthrough routes). Model load/unload, cluster-local calls (health/load/chat, peer-node requests) and deployment load/unload/metrics also go direct via `mlx_tool_url` with the MLX internal key — the gateway's `/v1/models/{id}/load|unload` intercept rewrites into the Hub's own `POST /api/v1/models/{id}/serve`, so routing Hub load/unload through the gateway 404s on unknown IDs and would recurse forever on known ones. `FMH_MLX_URL` env now overrides the CLI serve default (same P1-21 env-hook pattern as host/port). Verified E2E against a live gateway + MLX: health through the gateway (mlxConnected true), hardware/recommend via direct MLX, serve→load via direct MLX 200, chat through the gateway 200. (`server/config.py`, `server/__main__.py`, `server/routers/{inference,cluster,deployments,system,hardware,recommend,adapt,analyze,benchmarks,quantize}.py`, `server/tasks.py`, `cli/{main,recommend,analyze}.py`, `api/base_binding.py`, `tests/test_config.py`, `CLAUDE.md`, `README.md`, `README_CN.md`)

## [1.2.2] — 2026-10-24

Patch release: union-web shell module UI package (#56) + CI unblocks (MinIO image removal, identity tests).

### Added

- **#56 `web/` union-web shell module UI package (PRD N-4)** — model hub management console that mounts into the union-web shell at `/shell/fusion-models-hub/` or runs standalone. Stack per spec §10.1: React 18 + Vite 5 + TypeScript + Tailwind (fusion preset) + Bun, sharing `@fusion/web-ui` (DataTable/MetricTile/StatusBadge/KVPanel/ConfirmDialog/states) and `@fusion/web-sdk` via `file:vendor/@fusion`. Four pages over the REST API: **Models** (keyword search, detail panel with versions/tags, HF import via hf-mirror.com), **Downloads** (progress, client-derived speed/ETA, HF_MIRROR source badge, cancel behind danger confirm — spec §7.3/S5), **Quantize** (task submit + live status table), **Storage** (usage tiles + Fusion-MLX health). `web/manifest.json` per PRD §5.1 (id `fusion-models-hub`, category `platform`, devices desktop + mobile-ready); dev port 11475 registered in `architecture/port-registry.yaml`; four states on every data view; no auth logic in the UI (S8 — gateway injects `X-API-Key` embedded, `VITE_HUB_API_KEY` standalone dev). Verified E2E against a live server: real download created/completed through the UI, quantize task submitted through the UI, `?embedded=1` hides standalone nav; 34 unit tests, lint/typecheck/build green, bundle 56KB gzip. (`web/`, `README.md`, `README_CN.md`)

### Fixes

- **CI MinIO service start (both Docker Hub and quay.io pulls 401)** — MinIO removed all free community container images (2026 StorageSuite pivot): `minio/minio:latest` on Docker Hub returns "repository does not exist" and `quay.io/minio/minio` anonymous pulls are disabled. Switched the CI manual `docker run` and the pg-minio compose file to `cgr.dev/chainguard/minio` (same binary, env vars and `/minio/health/ready` endpoint; manifest pull verified 200): distroless `latest` + `--tmpfs /data:uid=65532,gid=65532` in CI, `latest-dev` (busybox shell) with a wget healthcheck in compose (the minimal image has no `mc`/shell). (`.github/workflows/ci.yml`, `tests/integration/docker-compose-pg-minio.yml`)
- **unit CI `test` job red on main since #54 (2026-09-03)** — `tests/test_identity_integration.py` builds the app with `identity_integration_enabled=True` and asserts the TenantMiddleware's fail-closed 401s, but unit CI installs only this package; `fusion-core` lives in a sibling repo (not on PyPI), so `fusion_core.tenant` cannot import and the app degrades to the local X-API-Key fallback, turning the assertions into 200s. The runtime fallback is intentional (`server/app.py` WARN); the tests now `pytest.importorskip("fusion_core.tenant")` so they run in full wherever fusion-core is installed (the monorepo venv — 11 tests locally) and skip in the unit CI env. (`tests/test_identity_integration.py`)

## [1.2.1] — 2026-09-04

Patch release: identity-aware combo mode (Authorization: Bearer + X-Tenant-Id alongside X-API-Key), completing the fusion-studio integration (#55).

### Added

- **#55 Accept `Authorization: Bearer` + `X-Tenant-Id` alongside `X-API-Key` (identity-aware combo mode)** — fusion-studio's `ModelHubAPIClient` attaches identity headers to all requests when the user is logged in via fusion-identity, in addition to the existing `X-API-Key`. In identity-aware mode (`FMH_IDENTITY_INTEGRATION_ENABLED=true`), the TenantMiddleware (installed in #54) already verifies the Bearer JWT and enforces the `tid`↔`X-Tenant-Id` match. This release closes the remaining gaps so the three-header combo is correct and safe: (1) `/api/v1/auth/keys` is no longer exempt from the tenant middleware in identity mode, so API-key provisioning receives the verified JWT `tid` and keys are minted tenant-scoped — previously every key was tenant-less in identity mode and the cross-tenant check had nothing to compare; (2) `auth_middleware` now rejects a local `X-API-Key` whose `tenant_id` does not match the JWT `tid` (401, "API key tenant does not match authenticated tenant"), blocking cross-tenant key reuse — `verify_api_key` has no tenant filter, so the match is enforced post-verify; a legacy tenant-less key is allowed through, only a concrete mismatch is rejected; (3) fusion-identity JWT roles (`tenant_admin`/`admin`/`developer`/`member`/`viewer`) are normalized onto the local `admin`/`developer`/`viewer` ladder so the create-key admin-or-bootstrap guard and tenanted-admin pinning work under identity roles — an unmapped role falls back to viewer (fail-closed). `X-API-Key`-only mode (identity disabled, the default) is unchanged. 3 new tests cover matching-tenant combo (200), cross-tenant key rejected (401), and X-API-Key-only preserved when identity off. (`server/identity.py`, `server/auth.py`, `server/routers/auth.py`, `tests/test_identity_integration.py`)

## [1.2.0] — 2026-09-03

Feature release: optional fusion-identity integration (multi-tenant PRD §3/§4).

### Added

- **#54 integrate fusion-identity: retire blind-trust of `X-Fusion-Tenant`** — config-gated security core. New `identity_integration_enabled` setting (env `FMH_IDENTITY_INTEGRATION_ENABLED`, default OFF — zero breakage to single-tenant dev / direct-CLI access). When ON, `install_tenant_middleware` from `fusion_core.tenant` is wired as the outermost middleware (last-added = runs first): it enforces `X-Tenant-Id` header presence (401 if missing), calls fusion-identity `POST /api/v1/auth/verify` (Bearer service token, `FUSION_IDENTITY_SERVICE_TOKEN`) to verify the inbound JWT, and rejects a forged `X-Tenant-Id` that does not match the JWT `tid` claim (401 on mismatch) — replacing the #53 blind-trust of the client-supplied `X-Fusion-Tenant`. The verified `tid` becomes the authoritative tenant; `role` is read from the verify response (not the local key row). The local `X-API-Key` path is retained only for hub-local model/module ACL policy keyed by the centrally-resolved tid; a request with a valid JWT but no local key passes through. Health/docs stay exempt so liveness probes keep working. Build decision is purely per-`Settings` (per-app `app.state.identity_integration_enabled`), never a module global — an identity-enabled test app cannot leak into another app's auth path under pytest-randomly reordering. Companion env: `FMH_IDENTITY_URL`, `FUSION_IDENTITY_SERVICE_TOKEN`, `FUSION_IDENTITY_JWT_ISSUER`, `FUSION_IDENTITY_JWT_AUDIENCE`. 8 tests (`tests/test_identity_integration.py`) mock fusion-identity HTTP: missing tenant header rejected, missing token rejected, forged tenant-header mismatch rejected, revoked/invalid token rejected, valid token + matching tenant allowed, role from verify response, health stays public, cross-tenant model access denied. Quota/usage/gRPC-lease deferred as follow-up. (`server/identity.py` new, `server/config.py`, `server/app.py`, `server/auth.py`)

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
