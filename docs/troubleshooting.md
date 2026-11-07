---
doc_id: fusion-models-hub/troubleshooting
product: fusion-models-hub
class: core
entry: service
doc_type: troubleshooting
title: { zh: "fusion-models-hub 故障排查", en: "fusion-models-hub troubleshooting" }
audience: engineer
status: draft
version: 1.2.15
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - docs/secret-rotation-runbook.md
  - start.sh
---

# fusion-models-hub 故障排查（Troubleshooting）

先收集：`./start.sh status`（进程/端口）、`curl -s localhost:11444/api/v1/system/health`、`logs/`。

## 症状速查

| 症状 | 原因 | 处置 |
|---|---|---|
| `401 {"detail":"API key required"}` | 鉴权默认开启（`FMH_AUTH_ENABLED=true`），请求未带 key | 经管理流程创建 key，请求带 `X-API-Key` 头 |
| `401 {"detail":"Invalid API key"}` | key 错误或已轮换；或误用网关管理令牌（`FUSION_ADMIN_KEY` 是网关/引导令牌，不通用） | 用 Hub 数据库内的有效 key；轮换流程见 [secret-rotation-runbook](secret-rotation-runbook.md) |
| `health.mlxConnected:false` | fusion-mlx / fusion-gateway 未运行，或 `FMH_MLX_URL` 指错 | 启动 MLX 栈；核对 `FMH_MLX_URL`（:11432 网关 / :11434 引擎）。仓库功能不受影响，仅转发端点不可用 |
| 启动报数据库连接错误 | PostgreSQL 未建库，或 URL 未规范化 | `createdb fusion_model_hub`；裸 `postgresql://` 会自动改写为 `postgresql+asyncpg://`，仍失败则检查 `DATABASE_URL` 是否被 `FMH_DB_URL` 之外的值覆盖 |
| 量化/LoRA 任务一直 pending | 任务为后台执行，大模型耗时长；或任务进程异常 | 先 `get_quantize_status` 看状态与错误详情；持续无进展查 `logs/` |
| 上传 413 | 超过 `FMH_MAX_UPLOAD_SIZE_MB`（默认 500） | 调大该值或改用模型目录路径注册（不传文件） |
| 端口被占 | 11444 被其他进程占用 | `./start.sh status` 看是否已有实例；端口注册见 `~/fusion/architecture/port-registry.yaml` |

## 排障开关

- `FMH_AUTH_ENABLED=false`：仅限本机复现鉴权问题时临时使用，验证完立即恢复 `true`。
- CORS 报错：核对 `FMH_CORS_ORIGINS` 是否包含前端来源（默认 `*` 已放宽，收紧后需显式列出）。

## 仍未解决

带 `health` 输出 + `logs/` 最后 50 行 + 复现命令提 issue（英文）。
