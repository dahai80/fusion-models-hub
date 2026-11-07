---
doc_id: fusion-models-hub/config-reference
product: fusion-models-hub
class: core
entry: service
doc_type: config-reference
title: { zh: "fusion-models-hub 配置参考", en: "fusion-models-hub configuration reference" }
audience: engineer
status: draft
version: 1.2.15
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - .env.example
  - fusion_model_hub/server/config.py
---

# fusion-models-hub 配置参考（Configuration Reference）

全部配置经环境变量（或 `.env`），无配置文件。前缀 `FMH_`（数据库/网关集成项除外）。

## 核心服务

| 变量 | 默认 | 说明 |
|---|---|---|
| `FMH_DATA_DIR` | `./data` | 数据目录（SQLite、上传文件等） |
| `FMH_MODELS_DIR` | `~/.fusion-mlx/models` | 模型文件根目录 |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/fmh.db` | 数据库连接；先读 `DATABASE_URL`，未设再读 `FMH_DB_URL`。裸 `postgresql://` 会规范化为 `postgresql+asyncpg://` |
| `FMH_AUTH_ENABLED` | `true` | API 鉴权开关；关闭只限本机调试 |
| `FMH_CORS_ORIGINS` | `*` | CORS 允许来源 |
| `FMH_MAX_UPLOAD_SIZE_MB` | `500` | 上传大小上限 |
| `FMH_TLS_CERTFILE` / `FMH_TLS_KEYFILE` | 空 | TLS 证书；两者都设才启用 |

## MLX / 网关连接

| 变量 | 默认 | 说明 |
|---|---|---|
| `FMH_MLX_URL` | `http://127.0.0.1:11432` | 推理主地址（fusion-gateway；直连引擎则为 :11434） |
| `FMH_MLX_TOOL_URL` | `http://127.0.0.1:11434` | 工具调用转发地址（fusion-mlx 引擎） |

`health.mlxConnected` 反映连接探测结果；未连接时仅 LLM 转发类端点不可用。

## 存储

| 变量 | 默认 | 说明 |
|---|---|---|
| `FMH_STORAGE_TYPE` | `local` | `local` 或 `minio` |
| `FMH_MINIO_ENDPOINT` / `FMH_MINIO_ACCESS_KEY` / `FMH_MINIO_SECRET_KEY` / `FMH_MINIO_BUCKET` | 空 | `minio` 模式必填 |
| `FMH_BACKUP_DIR` | 空 | 备份目录 |

## Web Shell 集成

| 变量 | 默认 | 说明 |
|---|---|---|
| `FUSION_SHELL_GATEWAY_URL` | 空 | web-shell 网关地址，用于自注册 |
| `FUSION_ADMIN_KEY` | 空 | 自注册用的管理令牌（**勿写入任何文档/提交**） |
| `FMH_WEB_DIST_DIR` | 空 | Web 控制台构建产物目录 |
| `FMH_SHELL_MANIFEST` | 空 | shell manifest 输出路径 |

## 注意事项

- `FUSION_ADMIN_KEY` 属于网关/引导令牌，Hub 自身 API key 存在 Hub 数据库内，两者不通用（见 [get-started](get-started.md)）。
- `FMH_AUTH_ENABLED=false` 仅限本机排障，见 [troubleshooting](troubleshooting.md)。
- 密钥轮换流程见 [secret-rotation-runbook](secret-rotation-runbook.md)。
