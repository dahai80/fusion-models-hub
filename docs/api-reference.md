---
doc_id: fusion-models-hub/api-reference
product: fusion-models-hub
class: core
entry: service
doc_type: api-reference
title: { zh: "fusion-models-hub API 参考", en: "fusion-models-hub API reference" }
audience: engineer
status: draft
version: 1.2.15
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - fusion_model_hub/sdk/client.py
  - fusion_model_hub/server/app.py
---

# fusion-models-hub API 参考（API Reference）

Base URL `http://127.0.0.1:11444`。除 `/api/v1/system/health` 外全部需要 `X-API-Key` 头。
无 key / 无效 key 返回 `401`：`{"detail":"API key required"}` / `{"detail":"Invalid API key"}`。
错误统一为 `{"detail": string}`。交互式文档：服务启动后访问 `/docs`（OpenAPI）。

## 系统端点

| 端点 | 说明 |
|---|---|
| `GET /api/v1/system/health` | 公开；版本、MLX 连接、模型数、存储水位 |

## 模型与版本（SDK 方法 ↔ 端点）

SDK 客户端 `fusion_model_hub.sdk.client.FusionModelHubClient` 与 REST 端点一一对应：

| 能力 | SDK 方法 | REST |
|---|---|---|
| 列表 / 详情 | `list_models` / `get_model` | `GET /api/v1/models`（分页/过滤） / `GET /api/v1/models/{id}` |
| 增删改 | `create_model` / `update_model` / `delete_model` / `batch_delete` | 对应 `POST/PATCH/DELETE /api/v1/models...` |
| HF 导入 / 同步 | `import_from_hf` / `sync_registry` | `POST .../import` / `POST .../sync` |
| 对比 / 基准 | `compare_models` / `benchmark` | 对比与基准记录端点 |
| 版本生命周期 | `list_versions` / `get_version` / `promote` / `rollback` / `deprecate` / `retire_version` | `.../versions` 子资源 |
| 量化 | `start_quantize` / `list_quantize_tasks` / `get_quantize_status` / `evaluate_quantize` | 异步任务：创建返回句柄，轮询状态 |
| 分层量化 | `start_layered_quantize` / `get_layered_quantize_job` / `list_layered_quantize_jobs` | 同上 |
| LoRA 合并 | `start_lora_merge` / `get_lora_merge_status` | 同上 |
| OpenAI 兼容 | `chat_completions` / `completions` | `POST /v1/chat/completions`、`POST /v1/completions` |

精确路径以 `/docs`（OpenAPI schema）为准——SDK `client.py` 与服务端路由同仓同版本，是权威映射。

## 调用示例（SDK）

```python
from fusion_model_hub.sdk.client import FusionModelHubClient

client = FusionModelHubClient(base_url="http://127.0.0.1:11444", api_key="<your-key>")
client.list_models()
client.start_quantize(model_id="<id>", bits=4)   # 返回任务句柄
client.get_quantize_status("<task-id>")          # 轮询
```

## 约定

- 列表端点分页；`limit=0` 语义以端点实现为准（有上限保护，见 `/docs` 参数说明）。
- 异步任务端点不阻塞：先 `start_*`，再 `get_*_status`；任务失败状态也经轮询返回，不抛同步异常。
- 转发端点（chat/completions）行为依赖 `FMH_MLX_URL` 连接状态；MLX 不可用时返回错误 detail，而非挂起。
