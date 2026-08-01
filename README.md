# Fusion Model Hub

[![CI](https://github.com/dahai80/fusion-models-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/dahai80/fusion-models-hub/actions/workflows/ci.yml)

English | [中文](README_CN.md)

Unified model repository and management center for the Fusion-MLX ecosystem on macOS Apple Silicon.

## Features

- **REST API Server** — FastAPI async server with full model lifecycle management
- **Model CRUD** — Create, list, search, recommend, update, delete models with tags
- **Version Management** — Upload model versions with file storage, SHA256 hash verification
- **Chunked Upload** — Support for large model files via chunked upload (5MB chunks)
- **HuggingFace Import** — Import model metadata from HuggingFace repos via HF Mirror API, with optional `download=true`
- **Download Tracking** — Download counting and file serving
- **Status Lifecycle** — Version state machine: draft → testing → published → deprecated → retired
- **Version Promote** — One-call promotion through the full lifecycle (DRAFT→TESTING→PUBLISHED) with webhook dispatch
- **Quantization** — Async quantize tasks (2/4/6/8-bit) via Fusion-MLX with task tracking + compare endpoint
- **LoRA Merge** — Async LoRA adapter merge tasks with quantization support
- **URL Download** — Download model versions from URL with async background processing (SSRF-protected)
- **MLX Health Check** — System health includes Fusion-MLX availability detection
- **RBAC** — Role-based access control (admin/developer/viewer) via API key roles
- **Multi-Tenant** — Tenant isolation with tenant_id on models, API keys, audit logs
- **Model Ownership** — Owner-based access control; only the creator can edit/delete their models (when auth enabled)
- **Webhooks** — Event notifications with HMAC-SHA256 signing and retry with exponential backoff
- **Deployments** — Model deployment tracking with Fusion-MLX load/unload integration
- **Gray Release** — Canary deployment with traffic ratio control routed via inference proxy
- **Scaling** — Deployment replica scaling with MLX load integration
- **Evaluation** — Benchmark evaluation tracking with per-version scoring and cross-version comparison
- **Search & Recommend** — Advanced model search (keyword, architecture, quantization, benchmark score) + recommendation engine
- **Differential Sync** — Push/pull model metadata between FMH instances with manifest-based versioning
- **Export/Import** — Offline data export/import (JSON + tar.gz with model files)
- **Security Scan** — Model/version security scanning (malicious code, unsafe dependencies, sensitive info detection)
- **Watermark** — Embed and verify model watermarks with SHA256 signature
- **Encryption** — AES-256 (Fernet) encryption/decryption for version files at rest
- **Approvals** — Multi-level approval workflow (L1 auto-approve, L2/L3 manual review) for version publishing
- **Git LFS** — Git LFS v2 batch API + lock management for large model files
- **Distributed Tasks** — Cluster-wide distributed task execution with node targeting
- **SDK Client** — Synchronous Python client (`FusionModelHubClient`) for all API endpoints
- **Storage Abstraction** — Pluggable storage backend (LocalStore + MinioStore)
- **CLI** — `serve`, `export`, `import`, `migrate` subcommands
- **Alembic Migrations** — Database schema migration support
- **Ratings** — Model rating system (1-5 score + comment) with average score aggregation
- **Favorites** — User favorites/bookmarks for models with duplicate prevention
- **Branches** — Model version branching (active/merged/archived) with merge operation
- **Evaluation Thresholds** — Enforce minimum benchmark scores before publishing (L1≥50, L2≥70, L3≥85)
- **Compliance Fields** — License type and data compliance tracking on model versions
- **Calibration Dataset** — Quantize tasks support calibration dataset specification
- **Hardware Detection** — Apple Silicon chip detection (M1-M5), VRAM/RAM profiling via Fusion-MLX with 5-min cache
- **Smart Recommendation** — Multi-dim scoring (hardware fit + quality + speed + popularity) with preference-based weight profiles, batch MLX evaluation
- **Adaptation Decision** — Migration level assessment (L0-L4), compile strategy, quantize suggestions, migration plans, execute pipeline (assess→convert→quantize)
- **3-Level Cache** — raw/ → converted/ → quantized/{bits}bit/ with index, GC, validation, MLX version awareness
- **Benchmarks** — Proxy MLX benchmark data with chip/model/quant filtering
- **Model Analysis** — Proxy MLX structure analysis (architecture, layers, params, special ops)
- **Layered Quantize** — Per-layer quantization with configurable bits, group size, and mode
- **CLI** — `fmh` typer-based CLI with download, recommend, list, analyze, hardware subcommands
- **Prometheus Metrics** — `/metrics` endpoint with request counters, duration histograms, active gauges
- **Auto Backup** — Configurable periodic JSON backup of models/versions to file
- **Task Recovery** — Pending quantize tasks auto-restart on server restart; orphaned running tasks marked failed
- **TLS** — HTTPS support via `--tls-certfile` and `--tls-keyfile` CLI flags
- **Async SDK** — `AsyncFusionModelHubClient` with httpx async support
- **Docker & Helm** — Dockerfile + Helm chart for Kubernetes deployment

## Quick Start

```bash
# Install
pip install -e ".[test]"

# Start the API server
fusion-model-hub serve --host 0.0.0.0 --port 8080

# Or with custom data directory
fusion-model-hub serve --data-dir /path/to/data --port 8080

# With TLS
fusion-model-hub serve --tls-certfile /path/to/cert.pem --tls-keyfile /path/to/key.pem

# Export data to JSON
fusion-model-hub export -o backup.json

# Import data from JSON
fusion-model-hub import -i backup.json

# Run database migrations
fusion-model-hub migrate --db-url sqlite+aiosqlite:///data/fmh.db
```

## API Endpoints

### Models

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models` | Create a model |
| GET | `/api/v1/models` | List models (keyword/type/arch filter, pagination) |
| GET | `/api/v1/models/{id}` | Get model detail with versions |
| PUT | `/api/v1/models/{id}` | Update model fields/tags (owner-only when auth enabled) |
| DELETE | `/api/v1/models/{id}` | Delete model and files (owner-only when auth enabled) |
| POST | `/api/v1/models/import/hf` | Import from HuggingFace repo (optional `download: true`) |
| GET | `/api/v1/models/search` | Advanced search (keyword, architecture, quantization, benchmark score) |
| GET | `/api/v1/models/recommend` | Recommend models by task type, model type, params size |

### Versions

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/versions` | Upload version (with optional file) |
| POST | `/api/v1/models/{id}/versions/chunk-upload` | Chunked upload for large files |
| GET | `/api/v1/models/{id}/versions` | List versions |
| GET | `/api/v1/versions/{id}` | Get version detail |
| PUT | `/api/v1/versions/{id}/status` | Change version status (lifecycle enforced) |
| GET | `/api/v1/versions/{id}/download` | Download version file |
| PUT | `/api/v1/versions/{id}/benchmark` | Update benchmark results |
| PUT | `/api/v1/versions/{id}/metrics` | Update version metrics |
| POST | `/api/v1/versions/{id}/promote` | Promote version lifecycle (DRAFT→TESTING→PUBLISHED) |
| POST | `/api/v1/versions/{id}/rollback` | Rollback to published |
| POST | `/api/v1/versions/{id}/deprecate` | Deprecate with optional successor |
| POST | `/api/v1/versions/{id}/retire` | Retire version |
| GET | `/api/v1/models/{id}/export` | Export model as tar.gz (metadata + files) |
| POST | `/api/v1/models/import-tar` | Import model from tar.gz upload |

### Evaluations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/evaluations` | Create evaluation |
| GET | `/api/v1/evaluations` | List evaluations (filter by model/version/benchmark/status) |
| GET | `/api/v1/evaluations/benchmarks/compare` | Compare benchmarks across versions |
| GET | `/api/v1/evaluations/{id}` | Get evaluation detail |
| PATCH | `/api/v1/evaluations/{id}` | Update evaluation (status/score/metrics) |
| DELETE | `/api/v1/evaluations/{id}` | Delete evaluation |

### Sync

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/sync/versions/{id}/manifest` | Get version file manifest |
| POST | `/api/v1/sync/push` | Push model to remote FMH instance |
| POST | `/api/v1/sync/pull` | Pull model from remote FMH instance |

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/system/health` | Health check (includes MLX status) |
| GET | `/api/v1/system/storage` | Storage statistics |
| GET | `/api/v1/system/audit` | Query audit logs |
| GET | `/api/v1/system/export` | Export all data (models, tenants, webhooks) |
| POST | `/api/v1/system/import` | Import data |

### Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/keys` | Create API key (with role) |
| GET | `/api/v1/auth/keys` | List API keys |
| DELETE | `/api/v1/auth/keys/{id}` | Delete API key |
| POST | `/api/v1/auth/keys/{id}/deactivate` | Deactivate API key |

### Tenants

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/tenants` | Create tenant |
| GET | `/api/v1/tenants` | List tenants |
| GET | `/api/v1/tenants/{id}` | Get tenant |
| PATCH | `/api/v1/tenants/{id}` | Update tenant |
| DELETE | `/api/v1/tenants/{id}` | Delete tenant |

### Webhooks

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/webhooks` | Create webhook |
| GET | `/api/v1/webhooks` | List webhooks |
| GET | `/api/v1/webhooks/{id}` | Get webhook |
| DELETE | `/api/v1/webhooks/{id}` | Delete webhook |

### Deployments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/deployments` | Create deployment (auto-loads model in MLX) |
| GET | `/api/v1/deployments` | List deployments |
| GET | `/api/v1/deployments/{id}` | Get deployment |
| PATCH | `/api/v1/deployments/{id}` | Update deployment |
| DELETE | `/api/v1/deployments/{id}` | Delete deployment (auto-unloads from MLX) |
| POST | `/api/v1/deployments/{id}/gray` | Enable gray release |
| DELETE | `/api/v1/deployments/{id}/gray` | Disable gray release |
| POST | `/api/v1/deployments/{id}/scale` | Scale replicas |
| GET | `/api/v1/deployments/{id}/metrics` | Get deployment metrics (MLX status + version) |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/serve` | Load model into Fusion-MLX |
| DELETE | `/api/v1/models/{id}/serve` | Unload model |
| GET | `/api/v1/models/{id}/serve` | Get serve status |
| POST | `/api/v1/inference/{id}/chat` | Chat completion (proxied, gray-release aware) |
| POST | `/api/v1/inference/{id}/completions` | Text completion (proxied) |
| POST | `/api/v1/inference/{id}/embeddings` | Embeddings (proxied) |

### Quantize

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/quantize` | Submit quantize task (2/4/6/8-bit) |
| GET | `/api/v1/quantize` | List quantize tasks |
| GET | `/api/v1/quantize/running` | List currently running tasks |
| GET | `/api/v1/quantize/{task_id}` | Get task status |
| GET | `/api/v1/quantize/{task_id}/compare` | Compare source vs quantized version metrics |
| POST | `/api/v1/quantize/layered` | Submit per-layer quantize task (per-layer bits, group size, mode) |
| GET | `/api/v1/quantize/layered/jobs` | List layered quantize jobs |
| GET | `/api/v1/quantize/layered/jobs/{job_id}` | Get layered quantize job status |

### URL Download

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/versions/download-url` | Download version from URL (async, SSRF-protected) |

### Cluster

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/cluster/nodes` | Add cluster node |
| GET | `/api/v1/cluster/nodes` | List cluster nodes |
| GET | `/api/v1/cluster/nodes/{id}` | Get node detail |
| DELETE | `/api/v1/cluster/nodes/{id}` | Remove node |
| POST | `/api/v1/cluster/nodes/{id}/heartbeat` | Node heartbeat |

### Batch & Sync

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/sync` | Sync registry from remote hub |
| POST | `/api/v1/models/batch/delete` | Batch delete models |
| POST | `/api/v1/models/batch/tag` | Batch tag models |
| GET | `/api/v1/models/compare` | Compare models (comma-separated IDs) |

### Security Scan

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/security/scan` | Trigger security scan for a model/version |
| GET | `/api/v1/security/scan/{scan_id}` | Get scan result by ID |
| GET | `/api/v1/security/scans` | List scans (filter by model_id, version_id, status) |

### Watermark

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/watermark/embed` | Embed watermark into a model/version |
| POST | `/api/v1/watermark/verify` | Verify watermark signature |
| GET | `/api/v1/watermark/list` | List watermarks (filter by model_id, version_id) |

### Encryption

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/encryption/encrypt` | Encrypt a version's file (AES-256 Fernet) |
| POST | `/api/v1/encryption/decrypt` | Decrypt a previously encrypted version file |
| GET | `/api/v1/encryption/status/{version_id}` | Check encryption status of a version |

### Approvals

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/approvals` | Submit approval request (L1 auto, L2/L3 manual) |
| GET | `/api/v1/approvals` | List approval requests (filter by model_id, status, level) |
| GET | `/api/v1/approvals/{req_id}` | Get approval request detail |
| POST | `/api/v1/approvals/{req_id}/approve` | Approve a pending request |
| POST | `/api/v1/approvals/{req_id}/reject` | Reject a pending request |

### Git LFS

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/gitlfs/objects/batch` | Git LFS v2 batch API (upload/download) |
| POST | `/api/v1/gitlfs/locks` | Create a lock on a model path |
| GET | `/api/v1/gitlfs/locks` | List locks (filter by model_id, path) |
| DELETE | `/api/v1/gitlfs/locks/{lock_id}` | Delete a lock |

### LoRA Merge

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/quantize/lora-merge` | Submit LoRA merge task (base + adapter, with quantization) |
| GET | `/api/v1/quantize/lora-merge/{task_id}` | Get LoRA merge task status |

### Distributed Tasks

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/cluster/distributed-tasks` | Submit distributed task with node targeting |
| GET | `/api/v1/cluster/distributed-tasks/{task_id}` | Get distributed task status |

### Ratings

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/ratings` | Create rating (score 1-5 + optional comment) |
| GET | `/api/v1/models/{id}/ratings` | List ratings (paginated, includes average_score) |
| GET | `/api/v1/models/{id}/ratings/summary` | Get average score + total count |
| DELETE | `/api/v1/ratings/{rating_id}` | Delete rating |

### Favorites

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/favorites` | Add model to favorites (409 if duplicate) |
| GET | `/api/v1/models/{id}/favorites` | List favorites for a model (paginated) |
| GET | `/api/v1/favorites/me` | List current user's favorites |
| DELETE | `/api/v1/favorites/{favorite_id}` | Remove favorite |

### Branches

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/branches` | Create branch |
| GET | `/api/v1/models/{id}/branches` | List branches (optional status filter) |
| GET | `/api/v1/branches/{branch_id}` | Get branch detail |
| PATCH | `/api/v1/branches/{branch_id}` | Update branch |
| DELETE | `/api/v1/branches/{branch_id}` | Delete branch |
| POST | `/api/v1/branches/{branch_id}/merge` | Merge branch (sets status to MERGED) |

### Prometheus Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Prometheus exposition format metrics |

### Hardware

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/hardware` | Get hardware profile (chip, VRAM, RAM, disk) |
| POST | `/api/v1/hardware/refresh` | Force refresh hardware detection cache |

### Recommend

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/recommend` | Get model recommendations (multi-dim scoring) |
| GET | `/api/v1/recommend/quick` | Quick recommendation (top 5) |

### Adapt

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/adapt/assess` | Assess model migration level (L0-L4) with source_format support |
| POST | `/api/v1/adapt/plan` | Generate migration plan with steps and quantize suggestion |
| POST | `/api/v1/adapt/execute` | Execute full adaptation pipeline (assess→convert→quantize) |
| GET | `/api/v1/adapt/execute/{execution_id}` | Get adaptation execution status |

### Benchmarks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/benchmarks` | List benchmarks (filter by chip, model_id, quant) |
| GET | `/api/v1/benchmarks/{model_id}` | Get best benchmark for model (filter by chip, quant) |

### Analyze

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/analyze` | Analyze model structure (architecture, layers, params, special ops) |

## SDK Client

The `FusionModelHubClient` provides a synchronous Python client for all API endpoints:

```python
from fusion_model_hub.sdk.client import FusionModelHubClient

client = FusionModelHubClient(base_url="http://localhost:8080", api_key="optional-key")

# Models
client.create_model({"name": "qwen2.5-7b", "model_type": "llm"})
client.list_models(keyword="qwen")
client.get_model("model-id")
client.update_model("model-id", {"description": "Updated"})
client.delete_model("model-id")
client.import_from_hf({"hf_repo": "Qwen/Qwen2.5-7B"})

# Versions
client.promote_version("version-id")
client.benchmark_version("version-id")
client.rollback_version("version-id")
client.deprecate_version("version-id")

# Quantize & LoRA
client.start_quantize("source-version-id", quant_bits=4, calibration_dataset="my-dataset")
client.start_lora_merge("base-version-id", "lora-version-id")

# Security & Watermark
client.start_security_scan("version-id")
client.embed_watermark("version-id", metadata='{"owner":"acme"}')
client.verify_watermark("version-id")

# Encryption
client.encrypt_version("version-id")
client.decrypt_version("version-id")

# Approvals
client.create_approval("version-id", level="L2", reason="Production release")
client.approve_request("req-id")
client.reject_request("req-id")

# Git LFS
client.gitlfs_batch("upload", [{"oid": "abc", "size": 1024}])
client.create_gitlfs_lock("models/qwen/safetensors")

# Cluster
client.add_node("node-1", "http://node1:8080")
client.submit_distributed_task("inference", "version-id", target_node_ids=["node-1"])

# Ratings
client.create_rating("model-id", score=5, comment="Excellent model")
client.list_ratings("model-id")
client.get_rating_summary("model-id")

# Favorites
client.add_favorite("model-id")
client.list_my_favorites()

# Branches
client.create_branch("model-id", name="experiment-v2")
client.list_branches("model-id")
client.merge_branch("branch-id")

# Hardware
client.get_hardware_info()
client.refresh_hardware()

# Recommend
client.recommend_models(task="llm", preference="speed", max_results=5)
client.quick_recommend(task="llm")

# Adapt
client.assess_model("model-id", hf_repo="org/model", source_format="safetensors")
client.plan_migration("model-id", params_b=7.0, hf_repo="org/model")
client.execute_adaptation("model-id", quant_bits=4, params_b=7.0)
client.get_adapt_execution("execution-id")

# Benchmarks
client.list_benchmarks(chip="M4 Pro", model_id="qwen2.5-7b")
client.get_benchmark("qwen2.5-7b", chip="M4 Pro", quant="4bit")

# Analyze
client.analyze_model(model_path="/path/to/model", hf_repo="org/model")

# Layered Quantize
client.start_layered_quantize("model-id", default_bits=4, layer_rules=[{"pattern": ".*lm_head", "bits": 8}])
client.get_layered_quantize_job("job-id")
client.list_layered_quantize_jobs()
```

### Async SDK Client

```python
from fusion_model_hub.sdk.async_client import AsyncFusionModelHubClient

async with AsyncFusionModelHubClient(base_url="http://localhost:8080") as client:
    models = await client.list_models()
    await client.create_rating("model-id", score=5)
```

## Example Usage

```bash
# Create a model
curl -X POST http://localhost:8080/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{"name": "qwen2.5-7b", "model_type": "llm", "architecture": "qwen2", "params_size": "7B"}'

# Upload a version with file
curl -X POST http://localhost:8080/api/v1/models/{model_id}/versions \
  -F "version=1.0.0" \
  -F "format=mlx" \
  -F "quantization=4bit" \
  -F "file=@model_weights.bin"

# Import from HuggingFace (metadata only)
curl -X POST http://localhost:8080/api/v1/models/import/hf \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-7B"}'

# Import from HuggingFace (with download)
curl -X POST http://localhost:8080/api/v1/models/import/hf \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-7B", "download": true}'

# Search models
curl "http://localhost:8080/api/v1/models/search?keyword=qwen&quantization=4bit&sort_by=benchmark_score"

# Get model recommendations
curl "http://localhost:8080/api/v1/models/recommend?task_type=text-generation&max_params=7B&limit=5"

# Submit quantize task
curl -X POST http://localhost:8080/api/v1/quantize \
  -H "Content-Type: application/json" \
  -d '{"source_version_id": "<version_id>", "quant_bits": 4}'

# Compare quantized vs source
curl "http://localhost:8080/api/v1/quantize/{task_id}/compare"

# Download version from URL
curl -X POST http://localhost:8080/api/v1/models/{model_id}/versions/download-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://hf-mirror.com/...", "version": "1.0.0-4bit"}'

# Export model as tar.gz
curl -o model.tar.gz "http://localhost:8080/api/v1/models/{model_id}/export"

# Create evaluation
curl -X POST http://localhost:8080/api/v1/evaluations \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "benchmark_name": "mmlu", "status": "running"}'

# Push model to remote FMH instance
curl -X POST http://localhost:8080/api/v1/sync/push \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "target_url": "https://other-fmh.example.com"}'

# Promote version through lifecycle (DRAFT→TESTING→PUBLISHED)
curl -X POST http://localhost:8080/api/v1/versions/{version_id}/promote

# Security scan
curl -X POST http://localhost:8080/api/v1/security/scan \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "scan_type": "full"}'

# Embed watermark
curl -X POST http://localhost:8080/api/v1/watermark/embed \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "metadata": "{\"owner\": \"acme\"}"}'

# Encrypt version file
curl -X POST http://localhost:8080/api/v1/encryption/encrypt \
  -H "Content-Type: application/json" \
  -d '{"version_id": "..."}'

# Submit approval request
curl -X POST http://localhost:8080/api/v1/approvals \
  -H "Content-Type: application/json" \
  -d '{"version_id": "...", "level": "L2", "reason": "Production release"}'

# LoRA merge
curl -X POST http://localhost:8080/api/v1/quantize/lora-merge \
  -H "Content-Type: application/json" \
  -d '{"base_version_id": "...", "lora_version_id": "...", "quant_bits": 4}'

# Submit distributed task
curl -X POST http://localhost:8080/api/v1/cluster/distributed-tasks \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "target_nodes": ["node-1", "node-2"]}'

# Assess model adaptation level
curl -X POST http://localhost:8080/api/v1/adapt/assess \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3.2-1b", "hf_repo": "meta-llama/Llama-3.2-1B", "source_format": "safetensors"}'

# Execute full adaptation pipeline
curl -X POST http://localhost:8080/api/v1/adapt/execute \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3.2-1b", "quant_bits": 4, "params_b": 1.0}'

# Get adaptation execution status
curl http://localhost:8080/api/v1/adapt/execute/{execution_id}

# List benchmarks
curl "http://localhost:8080/api/v1/benchmarks?chip=M4+Pro&model_id=qwen2.5-7b"

# Get best benchmark for model
curl "http://localhost:8080/api/v1/benchmarks/qwen2.5-7b?chip=M4+Pro&quant=4bit"

# Analyze model structure
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-7B"}'

# Submit layered quantize (per-layer bits)
curl -X POST http://localhost:8080/api/v1/quantize/layered \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b", "default_bits": 4, "layer_rules": [{"pattern": ".*lm_head", "bits": 8}], "quant_group_size": 64}'

# Get layered quantize job status
curl http://localhost:8080/api/v1/quantize/layered/jobs/{job_id}
```

## Architecture

```
fusion_model_hub/
├── db/
│   ├── models.py          # SQLAlchemy ORM: Model, ModelVersion, ModelTag, QuantizeTask, ApiKey, AuditLog, ClusterNode, Tenant, Webhook, Deployment, EvaluationResult, SecurityScan, Watermark, ApprovalRequest, GitLfsLock, LoraMergeTask, DistributedTask, ModelRating, ModelFavorite, ModelBranch
│   ├── database.py        # Async engine & session factory (aiosqlite)
│   └── crud.py            # Async CRUD operations with field whitelists
├── storage/
│   ├── base.py            # StorageBackend ABC
│   ├── local_store.py     # LocalStore: chunked upload, SHA256, assemble, encryption
│   └── minio_store.py     # MinioStore: S3-compatible object storage
├── server/
│   ├── app.py             # FastAPI app factory with lifespan + backup scheduler
│   ├── config.py          # Settings dataclass (env vars + MinIO + backup + TLS config)
│   ├── deps.py            # Dependency injection (Session, Store[Backend], Settings)
│   ├── auth.py            # Auth middleware with RBAC + owner enforcement
│   ├── tasks.py           # Async task manager (quantize tasks with calibration)
│   ├── backup.py          # Auto backup scheduler (configurable interval, JSON dump)
│   ├── metrics.py         # Prometheus metrics middleware + /metrics endpoint
│   ├── __main__.py        # CLI entry point (serve/export/import/migrate + TLS)
│   └── routers/
│       ├── models.py      # /api/v1/models + HF import + sync/batch/compare/search/recommend
│       ├── versions.py    # /api/v1/versions + lifecycle + promote + benchmark + metrics + tar export/import
│       ├── quantize.py    # /api/v1/quantize + compare + LoRA merge
│       ├── inference.py   # /api/v1/inference proxy + gray-release routing
│       ├── auth.py        # /api/v1/auth key management + RBAC roles
│       ├── cluster.py     # /api/v1/cluster nodes + heartbeat + distributed tasks
│       ├── system.py      # /api/v1/system (health + MLX + audit + export/import)
│       ├── tenants.py     # /api/v1/tenants CRUD
│       ├── webhooks.py    # /api/v1/webhooks + event dispatcher + retry
│       ├── deployments.py # /api/v1/deployments + gray release + scale + metrics + MLX integration
│       ├── evaluations.py # /api/v1/evaluations + benchmark compare
│       ├── sync.py        # /api/v1/sync (push/pull/manifest)
│       ├── security.py    # /api/v1/security scan
│       ├── watermark.py   # /api/v1/watermark embed/verify
│       ├── encryption.py  # /api/v1/encryption encrypt/decrypt/status
│       ├── approvals.py   # /api/v1/approvals submit/approve/reject
│       ├── gitlfs.py      # /api/v1/gitlfs batch + locks
│       ├── ratings.py     # /api/v1/models/{id}/ratings CRUD + summary
│       ├── favorites.py   # /api/v1/models/{id}/favorites + /me
│       ├── branches.py    # /api/v1/models/{id}/branches + merge
│       ├── hardware.py    # /api/v1/hardware (proxy MLX + refresh)
│       ├── recommend.py   # /api/v1/recommend (multi-dim scoring + batch MLX)
│       ├── adapt.py       # /api/v1/adapt (assess + plan + execute pipeline)
│       ├── benchmarks.py  # /api/v1/benchmarks (proxy MLX benchmarks)
│       └── analyze.py     # /api/v1/analyze (proxy MLX model structure analysis)
├── sdk/
│   ├── client.py          # FusionModelHubClient — synchronous Python SDK
│   ├── async_client.py    # AsyncFusionModelHubClient — async Python SDK
│   └── models.py          # Pydantic request/response models for SDK
├── api/
│   └── base_binding.py    # FusionMLX HTTP client
├── hardware/
│   ├── __init__.py        # Exports HardwareDetector, HardwareProfile
│   ├── types.py           # ChipGeneration, GPUProfile, CPUProfile, HardwareProfile
│   └── detector.py        # HardwareDetector — MLX hardware detection with cache
├── recommend/
│   ├── __init__.py        # Exports RecommendEngine, ModelRecommendation
│   ├── types.py           # RecommendRequest, ModelRecommendation, RecommendResponse
│   ├── scorer.py          # Multi-dim scoring (hw fit, quality, speed, popularity)
│   └── engine.py          # RecommendEngine — batch MLX recommend + scorer fallback
├── adapt/
│   ├── __init__.py        # Exports AdaptDecisionEngine, AdaptationLevel
│   ├── types.py           # AdaptationLevel (L0-L4), MigrationPlan, AdaptationResult
│   ├── migration.py       # Migration plan generation + quantize suggestions
│   └── decision.py        # AdaptDecisionEngine — MLX migration-level + analyze enrichment + local fallback
├── cache/
│   ├── __init__.py        # Exports CacheManager, CacheLevel
│   ├── types.py           # CacheLevel, CacheEntry, CacheStats
│   └── manager.py         # 3-level cache (raw/converted/quantized) + GC + validate + MLX version awareness
├── cli/
│   ├── __init__.py        # Exports typer app
│   ├── main.py            # fmh CLI entry (hardware, version, sub-apps)
│   ├── download.py        # fmh download (hf, url)
│   ├── recommend.py       # fmh recommend (models, quick)
│   ├── list_cmd.py        # fmh list (local, remote, stats)
│   └── analyze.py         # fmh analyze (assess, plan)
├── convert/
│   └── converter.py       # Model conversion via Fusion-MLX
├── manage/
│   └── manager.py         # Local model manager
└── repo/
    ├── models.py           # Data models (ModelInfo)
    ├── registry.py         # In-memory model catalog
    └── downloader.py       # Async download with resume
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FMH_DATA_DIR` | `./data` | Data directory for DB and files |
| `FMH_MLX_URL` | `http://localhost:11434` | Fusion-MLX server URL |
| `FMH_AUTH_ENABLED` | `false` | Enable API key authentication |
| `FMH_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `FMH_MAX_UPLOAD_SIZE_MB` | `500` | Max upload file size |
| `FMH_DB_URL` | `sqlite+aiosqlite:///data/fmh.db` | Database URL (supports PostgreSQL) |
| `FMH_ALEMBIC_URL` | `sqlite://` | Sync DB URL for Alembic migrations |
| `FMH_STORAGE_TYPE` | `local` | Storage backend type (`local` or `minio`) |
| `FMH_MINIO_ENDPOINT` | `` | MinIO endpoint (when storage_type=minio) |
| `FMH_MINIO_ACCESS_KEY` | `` | MinIO access key |
| `FMH_MINIO_SECRET_KEY` | `` | MinIO secret key |
| `FMH_MINIO_BUCKET` | `fusion-models` | MinIO bucket name |
| `FMH_MINIO_SECURE` | `true` | Use HTTPS for MinIO |
| `FMH_BACKUP_DIR` | `` | Directory for auto-backup JSON files |
| `FMH_TLS_CERTFILE` | `` | TLS certificate file path |
| `FMH_TLS_KEYFILE` | `` | TLS private key file path |

CLI options override env vars: `--host`, `--port`, `--data-dir`, `--db-url`, `--mlx-url`, `--log-level`, `--tls-certfile`, `--tls-keyfile`

## CLI

The `fmh` command provides a typer-based CLI:

```bash
# View hardware profile
fmh hardware

# Download model from HuggingFace mirror
fmh download hf mlx-community/Llama-3.2-1B-Instruct-4bit --mirror https://hf-mirror.com

# Download from direct URL
fmh download url https://example.com/model.mlx my-model

# Get model recommendations
fmh recommend models --task llm --preference speed --max-results 5
fmh recommend quick --task llm

# List local models
fmh list local
fmh list remote --limit 20
fmh list stats

# Analyze model adaptation
fmh analyze assess mlx-community/Llama-3.2-1B-Instruct-4bit
fmh analyze plan mlx-community/Llama-3.2-1B-Instruct-4bit

# Version
fmh version
```

## Development

```bash
source .venv/bin/activate
pip install -e ".[test]"

# Run all tests
pytest

# Run API integration tests only
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=fusion_model_hub --cov-report=term-missing

# Start Fusion-MLX (for integration tests)
~/claude-home/fusion-mlx/start.sh start
```

## Model Download Mirror

Use `https://hf-mirror.com` for downloading models in regions with limited HuggingFace access.
