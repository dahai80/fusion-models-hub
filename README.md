# Fusion Model Hub

Unified model repository and management center for the Fusion-MLX ecosystem on macOS Apple Silicon.

## Features

- **REST API Server** — FastAPI async server with full model lifecycle management
- **Model CRUD** — Create, list, search, update, delete models with tags
- **Version Management** — Upload model versions with file storage, SHA256 hash verification
- **Chunked Upload** — Support for large model files via chunked upload (5MB chunks)
- **HuggingFace Import** — Import model metadata from HuggingFace repos via HF Mirror API
- **Download Tracking** — Download counting and file serving
- **Status Lifecycle** — Version state machine: draft → testing → published → deprecated → retired
- **Quantization** — Async quantize tasks (2/4/6/8-bit) via Fusion-MLX with task tracking
- **URL Download** — Download model versions from URL with async background processing
- **MLX Health Check** — System health includes Fusion-MLX availability detection

## Quick Start

```bash
# Install
pip install -e ".[test]"

# Start the API server
fusion-model-hub --host 0.0.0.0 --port 8080

# Or with custom data directory
fusion-model-hub --data-dir /path/to/data --port 8080
```

## API Endpoints

### Models

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models` | Create a model |
| GET | `/api/v1/models` | List models (keyword/type/arch filter, pagination) |
| GET | `/api/v1/models/{id}` | Get model detail with versions |
| PUT | `/api/v1/models/{id}` | Update model fields/tags |
| DELETE | `/api/v1/models/{id}` | Delete model and files |
| POST | `/api/v1/models/import/hf` | Import from HuggingFace repo |

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
| POST | `/api/v1/versions/{id}/rollback` | Rollback to published |
| POST | `/api/v1/versions/{id}/deprecate` | Deprecate with optional successor |
| POST | `/api/v1/versions/{id}/retire` | Retire version |

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/system/health` | Health check (includes MLX status) |
| GET | `/api/v1/system/storage` | Storage statistics |
| GET | `/api/v1/system/audit` | Query audit logs |

### Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/keys` | Create API key |
| GET | `/api/v1/auth/keys` | List API keys |
| DELETE | `/api/v1/auth/keys/{id}` | Delete API key |
| POST | `/api/v1/auth/keys/{id}/deactivate` | Deactivate API key |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/serve` | Load model into Fusion-MLX |
| DELETE | `/api/v1/models/{id}/serve` | Unload model |
| GET | `/api/v1/models/{id}/serve` | Get serve status |
| POST | `/api/v1/inference/{id}/chat` | Chat completion (proxied) |
| POST | `/api/v1/inference/{id}/completions` | Text completion (proxied) |
| POST | `/api/v1/inference/{id}/embeddings` | Embeddings (proxied) |

### Quantize

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/quantize` | Submit quantize task (2/4/6/8-bit) |
| GET | `/api/v1/quantize` | List quantize tasks |
| GET | `/api/v1/quantize/running` | List currently running tasks |
| GET | `/api/v1/quantize/{task_id}` | Get task status |

### URL Download

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/models/{id}/versions/download-url` | Download version from URL (async) |

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

# Import from HuggingFace
curl -X POST http://localhost:8080/api/v1/models/import/hf \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-7B"}'

# Submit quantize task
curl -X POST http://localhost:8080/api/v1/quantize \
  -H "Content-Type: application/json" \
  -d '{"source_version_id": "<version_id>", "quant_bits": 4}'

# Download version from URL
curl -X POST http://localhost:8080/api/v1/models/{model_id}/versions/download-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://hf-mirror.com/...", "version": "1.0.0-4bit"}'

# Search models
curl "http://localhost:8080/api/v1/models?keyword=qwen&model_type=llm&page=1&page_size=10"
```

## Architecture

```
fusion_model_hub/
├── db/
│   ├── models.py          # SQLAlchemy ORM: Model, ModelVersion, ModelTag, QuantizeTask, ApiKey, AuditLog, ClusterNode
│   ├── database.py        # Async engine & session factory (aiosqlite)
│   └── crud.py            # Async CRUD operations
├── storage/
│   └── local_store.py     # File storage: chunked upload, SHA256, assemble
├── server/
│   ├── app.py             # FastAPI app factory with lifespan
│   ├── config.py          # Settings dataclass (env vars)
│   ├── deps.py            # Dependency injection (Session, Store, Settings)
│   ├── tasks.py           # Async task manager (quantize tasks)
│   ├── __main__.py        # CLI entry point (uvicorn)
│   └── routers/
│       ├── models.py      # /api/v1/models + HF import + sync/batch/compare
│       ├── versions.py    # /api/v1/versions + lifecycle + benchmark
│       ├── quantize.py    # /api/v1/quantize endpoints
│       ├── inference.py   # /api/v1/inference proxy
│       ├── auth.py        # /api/v1/auth key management
│       ├── cluster.py     # /api/v1/cluster nodes
│       └── system.py      # /api/v1/system (health + MLX + audit)
├── api/
│   └── base_binding.py    # FusionMLX HTTP client
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

CLI options override env vars: `--host`, `--port`, `--data-dir`, `--db-url`, `--mlx-url`, `--log-level`

## Development

```bash
source .venv/bin/activate
pip install -e ".[test]"

# Run all tests
pytest

# Run API integration tests only
pytest tests/test_api.py -v

# Start Fusion-MLX (for integration tests)
~/claude-home/fusion-mlx/start.sh start
```

## Model Download Mirror

Use `https://hf-mirror.com` for downloading models in regions with limited HuggingFace access.
