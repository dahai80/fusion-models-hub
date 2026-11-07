---
doc_id: fusion-models-hub/architecture
product: fusion-models-hub
class: core
entry: service
doc_type: architecture
title: { zh: "fusion-models-hub 架构", en: "fusion-models-hub architecture" }
audience: engineer
status: draft
version: 1.2.15
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - fusion_model_hub/server/app.py
  - fusion_model_hub/db/models.py
  - ARCHITECTURE_COMPLIANCE.md
---

# fusion-models-hub 架构（Architecture）

## 模块布局

```text
fusion_model_hub/
  server/       FastAPI 应用：app.py（路由装配）、config.py（env 配置）、
                auth.py（X-API-Key 鉴权）、identity.py、tasks.py（后台任务）
  db/           SQLAlchemy async ORM（models.py），aiosqlite / asyncpg
  sdk/          FusionModelHubClient：给下游 Python 调用方的同步客户端
  storage/      local_store / minio_store：模型文件存储后端
  cli.py        fusion-model-hub 命令（serve 等）
```

## 运行时拓扑

```text
web-shell / SDK 客户端
        │  HTTP :11444 + X-API-Key
        ▼
fusion-models-hub (FastAPI)
   ├── db: SQLite(默认) / PostgreSQL(asyncpg)
   ├── storage: local(FMH_MODELS_DIR) / MinIO
   ├── tasks: 量化 / LoRA 合并 / 评估（后台任务，状态可查）
   └── LLM 转发 ──► FMH_MLX_URL (fusion-gateway :11432)
                └─► FMH_MLX_TOOL_URL (fusion-mlx :11434)
```

## 关键决策

- **鉴权在 Hub 自身**：API key 存 Hub 数据库（`auth.py`），不经 fusion-identity 会话，也不复用网关管理令牌。理由：模型管理操作是机器对机器场景，key 可独立轮换（见 secret-rotation-runbook）。
- **数据库双栈**：开发默认 SQLite（aiosqlite），生产可切 PostgreSQL；`DATABASE_URL` 优先于 `FMH_DB_URL`，`postgresql://` 自动规范化为 `postgresql+asyncpg://`，减少驱动配置错误。
- **任务异步化**：量化/LoRA 合并/评估均为后台任务，创建即返回任务句柄，`get_*_status` 轮询。理由：这些操作以分钟/小时计，同步 HTTP 会超时。
- **转发而非代理网关**：Hub 只转发 OpenAI 兼容调用到 MLX 栈，不做路由/重写；路由职责归 fusion-gateway。

## 依赖关系

- 上游依赖：fusion-core（HTTP/配置/日志底座，git tag 安装）；MLX 栈通过 HTTP 探测与转发，弱耦合。
- 下游消费：fusion-web-shell（自注册 + SDK）、运维脚本与 CLI。
- 边界：不修改其他 fusion-* 工程；对 MLX 栈不可用有明确降级（仓库功能可用，转发不可用）。

## 一致性

`ARCHITECTURE_COMPLIANCE.md` 记录与全仓架构约束的对齐检查结果。
