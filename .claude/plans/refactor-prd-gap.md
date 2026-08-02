# Fusion-Model-Hub PRD Gap Refactoring Plan

## Overview
Based on PRD (`fusion-model-hub-prd-ar.plan`) vs current implementation gap analysis, this plan covers: bug fixes, schema additions, new integrations, stub replacements, and upstream/downstream issue filing.

---

## Phase 1: Bug Fixes (Immediate, No Dependencies)

### 1.1 Fix inference.py MLX API paths (CRITICAL BUG)
- **Problem**: `inference.py` lines 48, 87, 116 call `/api/load` and `/api/unload` — these don't exist on fusion-mlx
- **Correct paths**: `/v1/models/{model_id}/load` and `/v1/models/{model_id}/unload` (confirmed in MLX server.py lines 764, 803; also correctly used in `deployments.py`)
- **Fix**: Replace all 3 occurrences in `_cleanup_loaded_models()`, `serve_model()`, `unload_model()`
- **Payload change**: `/api/load` sends `{"model": name}`, but `/v1/models/{model_id}/load` uses path param + optional body; update request format accordingly

### 1.2 Fix app.py version string
- **Problem**: Hardcoded `"0.1.0"` in `create_app()` (line 102), pyproject.toml says `1.0.1`
- **Fix**: Read version from `importlib.metadata` or `fusion_model_hub.__version__`

---

## Phase 2: Schema Changes (DB Model + CRUD)

### 2.1 Add `model_modules` field to Model
- **PRD requirement**: "单模型绑定指定可用上层 Fusion 模块，禁止越权调用" (per-model Fusion-module binding: Chat☑ Code☑ Design☐)
- **Change**: Add `model_modules: Mapped[str]` column to `Model` (comma-separated: "chat,code,rag,design,agent")
- **Default**: `""` (empty = all modules allowed, for backward compatibility)
- **CRUD**: Add `model_modules` to `_MODEL_UPDATABLE` whitelist
- **New API**: `PUT /api/v1/models/{model_id}/modules` — set allowed modules

### 2.2 Add `qps_limit` field to ApiKey
- **PRD requirement**: "API密钥独立限流" (per-key QPS rate limiting)
- **Change**: Add `qps_limit: Mapped[int]` column to `ApiKey` (default 0 = unlimited)
- **CRUD**: Add `qps_limit` to API key create/update schemas
- **Enforcement**: In `auth.py` middleware, check QPS via in-memory sliding window counter per key

### 2.3 Add `source` enum value MODELSCOPE
- **PRD requirement**: "HF+ModelScope + 私有仓库 + 本地目录四源聚合"
- **Change**: Add `MODELSCOPE = "modelscope"` to `ModelSource` enum

### 2.4 Add `idle_timeout_minutes` field to Model
- **PRD requirement**: Per-model idle timeout config ("闲置10分钟自动卸载")
- **Change**: Add `idle_timeout_minutes: Mapped[int]` to `Model` (default 60, 0 = never unload)
- **Impact**: Replace hardcoded `_LOADED_TTL = 3600` in inference.py with per-model timeout

---

## Phase 3: Multi-Source Market (Module 1)

### 3.1 ModelScope search integration
- **New file**: `fusion_model_hub/repo/modelscope_search.py`
- **API**: Search ModelScope models via their public API (`https://modelscope.cn/api/v1/models`)
- **Mirror**: Use `https://modelscope.cn` directly (China-hosted, no mirror needed)
- **Endpoint**: Extend `GET /api/v1/models/market/search` with `source` filter (hf/modelscope/all)

### 3.2 Model download from ModelScope
- **Extend**: `fusion_model_hub/repo/downloader.py` to support ModelScope download URLs
- **Endpoint**: Add `source` param to model import endpoint

---

## Phase 4: Cluster Integration (Module 4 — Replace Stubs)

### 4.1 Replace distributed task stub with fusion-multi-nodes integration
- **Current**: `cluster.py._run_distributed()` just does `asyncio.sleep(0.1)` — FAKE
- **Real integration**: Call fusion-multi-nodes API to:
  1. `POST /api/tasks/submit` — submit model sync task to target nodes
  2. `GET /api/cluster/stats` — poll node resource status
  3. `GET /api/kv/find/{model_name}` — check if model exists on node
- **File issue on fusion-multi-nodes**: Need model file sync API (currently only has task/heartbeat/routing)

### 4.2 Add model sync-to-cluster endpoint
- **New endpoint**: `POST /api/v1/cluster/sync-model`
- **Logic**: Find model file path → call multi-nodes `/api/tasks/submit` with model copy task
- **Fallback**: If multi-nodes unavailable, return 503 with clear message

### 4.3 Cross-node inference routing
- **New endpoint**: `POST /api/v1/cluster/route-inference`
- **Logic**: Check local memory → if overloaded, call multi-nodes `/api/routing/strategy` → route to idle node
- **Proxy**: Forward inference request to target node's fusion-mlx

---

## Phase 5: Fusion-Bench Integration (Module 6)

### 5.1 Auto-trigger bench on model updates
- **Hook**: After quantize task completes (`tasks.py`), check if bench auto-trigger is enabled
- **Call**: `POST http://{bench_url}/api/v1/tasks` with appropriate suite config
- **Webhook**: Bench results callback writes to `EvaluationResult` table
- **File issue on fusion-bench**: Need webhook/callback mechanism for result notification

### 5.2 Bench manual trigger endpoint
- **Already exists partially**: `benchmarks.py` proxies to MLX `/v1/benchmarks`
- **Enhancement**: Add `POST /api/v1/benchmarks/trigger` that calls fusion-bench directly
- **Config**: Add `FMH_BENCH_URL` to Settings (default: `http://localhost:8090`)

### 5.3 Bench result storage
- **Store**: Write bench results to `EvaluationResult` table (already exists in schema)
- **Display**: Add `GET /api/v1/models/{model_id}/benchmarks` endpoint

---

## Phase 6: QPS Rate Limiting Enforcement

### 6.1 In-memory sliding window rate limiter
- **New file**: `fusion_model_hub/server/rate_limit.py`
- **Implementation**: Dict of `{key_prefix: [(timestamp, count), ...]}`, prune entries older than 60s
- **Check**: In `auth_middleware`, after key validation, check QPS limit

### 6.2 Module-based access control
- **New header**: `X-Fusion-Module` (values: chat, code, design, rag, agent)
- **Check**: In `auth_middleware` or inference endpoints, verify module is in model's `model_modules`
- **Deny**: Return 403 if module not allowed for this model

---

## Phase 7: Upstream/Downstream Issues

### 7.1 Issue: fusion-mlx — Model info API for multi-source search
- **Need**: `/v1/model-info?repo_id=xxx&source=modelscope` endpoint
- **Purpose**: Hub needs to get model metadata from ModelScope for market search
- **Alternative**: Hub calls ModelScope API directly (no MLX dependency needed)

### 7.2 Issue: fusion-multi-nodes — Model file sync API
- **Need**: API to sync model files between nodes (currently only task/heartbeat APIs)
- **Specific**: `POST /api/sync/push` and `GET /api/sync/status/{task_id}`
- **Blocker**: Phase 4.1 and 4.2 depend on this

### 7.3 Issue: fusion-bench — Webhook callback for auto-trigger
- **Need**: Bench task completion webhook (POST result to callback_url)
- **Specific**: Add `callback_url` param to `POST /api/v1/tasks`
- **Blocker**: Phase 5.1 depends on this

### 7.4 Issue: fusion-gateway — Model-aware routing
- **Need**: Gateway should route based on model type (Chat→chat models, Code→code models)
- **Integration**: Gateway reads model modules from Hub API, routes accordingly
- **Not a blocker**: Can work without it (Hub does its own module check)

---

## Phase 8: Test Updates

### 8.1 Update inference tests for corrected MLX paths
- Fix test mocks from `/api/load` to `/v1/models/{model_id}/load`
- Verify load/unload/serve flow works with correct paths

### 8.2 Add tests for new schema fields
- `model_modules`, `qps_limit`, `idle_timeout_minutes`
- CRUD operations for new fields

### 8.3 Add tests for QPS rate limiting
- Request within limit → 200
- Request over limit → 429

### 8.4 Add tests for module access control
- Allowed module → 200
- Disallowed module → 403

---

## Execution Order

1. **Phase 1** (Bug fixes) — immediate, no dependencies
2. **Phase 2** (Schema) — after Phase 1, DB migration
3. **Phase 7** (File issues) — can start in parallel with Phase 2
4. **Phase 6** (QPS + module control) — depends on Phase 2
5. **Phase 3** (ModelScope) — independent, can start after Phase 2
6. **Phase 4** (Cluster) — depends on fusion-multi-nodes issue resolution
7. **Phase 5** (Bench) — depends on fusion-bench issue resolution
8. **Phase 8** (Tests) — continuous, each phase adds its own tests
