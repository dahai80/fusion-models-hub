# Fusion Model Hub

[![CI](https://github.com/dahai80/fusion-models-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/dahai80/fusion-models-hub/actions/workflows/ci.yml)

[English](README.md) | 中文

macOS Apple Silicon 上 Fusion-MLX 生态的统一模型仓库与管理中心。

## 特性

- **REST API 服务** — FastAPI 异步服务，完整的模型生命周期管理
- **模型 CRUD** — 创建、列表、搜索、推荐、更新、删除模型，支持标签
- **版本管理** — 上传模型版本，文件存储，SHA256 哈希校验
- **分块上传** — 大文件分块上传（5MB 分片）
- **HuggingFace 导入** — 通过 HF Mirror API 导入 HuggingFace 仓库元数据，可选 `download=true`
- **下载追踪** — 下载计数与文件服务
- **状态生命周期** — 版本状态机：draft → testing → published → deprecated → retired
- **版本晋升** — 一键晋升全生命周期（DRAFT→TESTING→PUBLISHED），支持 Webhook 派发
- **量化** — 异步量化任务（2/4/6/8-bit），通过 Fusion-MLX 执行，支持任务追踪与对比
- **LoRA 合并** — 异步 LoRA 适配器合并任务，支持量化
- **URL 下载** — 从 URL 下载模型版本，异步后台处理（SSRF 防护）
- **MLX 健康检查** — 系统健康状态包含 Fusion-MLX 可用性检测
- **RBAC** — 基于 API Key 角色的访问控制（admin/developer/viewer）
- **多租户** — 基于 tenant_id 的模型、API Key、审计日志租户隔离
- **模型所有权** — 基于所有者的访问控制；仅创建者可编辑/删除模型（开启认证时）
- **Webhooks** — 事件通知，HMAC-SHA256 签名，指数退避重试
- **部署** — 模型部署追踪，集成 Fusion-MLX 加载/卸载
- **灰度发布** — 金丝雀部署，流量比例控制，通过推理代理路由
- **弹性伸缩** — 部署副本伸缩，集成 MLX 加载
- **评测** — 基准评测追踪，按版本评分，跨版本对比
- **搜索与推荐** — 高级模型搜索（关键词、架构、量化、基准评分）+ 推荐引擎
- **差量同步** — FMH 实例间推送/拉取模型元数据，基于清单的版本管理
- **导出/导入** — 离线数据导出/导入（JSON + tar.gz 含模型文件）
- **安全扫描** — 模型/版本安全扫描（恶意代码、不安全依赖、敏感信息检测）
- **水印** — 嵌入和验证模型水印，SHA256 签名
- **加密** — AES-256 (Fernet) 加密/解密版本文件
- **审批** — 多级审批工作流（L1 自动批准，L2/L3 人工审核）
- **Git LFS** — Git LFS v2 批量 API + 锁管理
- **分布式任务** — 集群范围分布式任务执行，支持节点定向
- **SDK 客户端** — 同步 Python 客户端（`FusionModelHubClient`），覆盖全部 API
- **存储抽象** — 可插拔存储后端（LocalStore + MinioStore）
- **Alembic 迁移** — 数据库 Schema 迁移支持
- **评分** — 模型评分系统（1-5 分 + 评论），平均分聚合
- **收藏** — 用户收藏/书签，防重复
- **分支** — 模型版本分支（active/merged/archived），支持合并
- **评测阈值** — 发布前强制最低基准评分（L1≥50, L2≥70, L3≥85）
- **合规字段** — 许可证类型与数据合规追踪
- **校准数据集** — 量化任务支持校准数据集指定
- **硬件检测** — Apple Silicon 芯片检测（M1-M5），VRAM/RAM 分析，5 分钟缓存
- **智能推荐** — 多维评分（硬件适配 + 质量 + 速度 + 热度），偏好权重配置，批量 MLX 评估
- **适配决策** — 迁移等级评估（L0-L4），编译策略，量化建议，迁移方案，执行流水线（assess→convert→quantize）
- **三级缓存** — raw/ → converted/ → quantized/{bits}bit/，索引、GC、校验，MLX 版本感知
- **基准数据** — 代理 MLX 基准数据，支持芯片/模型/量化过滤
- **模型分析** — 代理 MLX 结构分析（架构、层、参数、特殊算子）
- **分层量化** — 按层量化，可配置 bits、group size、mode
- **CLI** — `fmh` typer CLI，支持 download、recommend、list、analyze、hardware 子命令
- **Prometheus 指标** — `/metrics` 端点，请求计数、耗时直方图、活跃指标
- **自动备份** — 可配置周期性 JSON 备份
- **任务恢复** — 服务重启自动重启待处理量化任务；孤立运行中任务标记为失败
- **TLS** — 通过 `--tls-certfile` 和 `--tls-keyfile` 支持 HTTPS
- **异步 SDK** — `AsyncFusionModelHubClient`，httpx 异步支持
- **Docker & Helm** — Dockerfile + Helm chart，支持 Kubernetes 部署

## 快速开始

```bash
# 安装
pip install -e ".[test]"

# 启动 API 服务
fusion-model-hub serve --host 127.0.0.1 --port 11444

# 自定义数据目录
fusion-model-hub serve --data-dir /path/to/data --port 11444

# 启用 TLS
fusion-model-hub serve --tls-certfile /path/to/cert.pem --tls-keyfile /path/to/key.pem

# 导出数据为 JSON
fusion-model-hub export -o backup.json

# 从 JSON 导入数据
fusion-model-hub import -i backup.json

# 运行数据库迁移
fusion-model-hub migrate --db-url sqlite+aiosqlite:///data/fmh.db
```

## API 端点

### 模型

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models` | 创建模型 |
| GET | `/api/v1/models` | 模型列表（关键词/类型/架构过滤，分页） |
| GET | `/api/v1/models/{id}` | 获取模型详情（含版本） |
| PUT | `/api/v1/models/{id}` | 更新模型字段/标签（开启认证时仅所有者） |
| DELETE | `/api/v1/models/{id}` | 删除模型及文件（开启认证时仅所有者） |
| POST | `/api/v1/models/import/hf` | 从 HuggingFace 仓库导入（可选 `download: true`） |
| GET | `/api/v1/models/search` | 高级搜索（关键词、架构、量化、基准评分） |
| GET | `/api/v1/models/recommend` | 按任务类型/模型类型/参数量推荐 |

### 版本

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models/{id}/versions` | 上传版本（可选文件） |
| POST | `/api/v1/models/{id}/versions/chunk-upload` | 大文件分块上传 |
| GET | `/api/v1/models/{id}/versions` | 版本列表 |
| GET | `/api/v1/versions/{id}` | 获取版本详情 |
| PUT | `/api/v1/versions/{id}/status` | 变更版本状态（生命周期约束） |
| GET | `/api/v1/versions/{id}/download` | 下载版本文件 |
| PUT | `/api/v1/versions/{id}/benchmark` | 更新基准结果 |
| PUT | `/api/v1/versions/{id}/metrics` | 更新版本指标 |
| POST | `/api/v1/versions/{id}/promote` | 晋升版本生命周期（DRAFT→TESTING→PUBLISHED） |
| POST | `/api/v1/versions/{id}/rollback` | 回滚至已发布 |
| POST | `/api/v1/versions/{id}/deprecate` | 废弃（可指定后继版本） |
| POST | `/api/v1/versions/{id}/retire` | 退役版本 |
| GET | `/api/v1/models/{id}/export` | 导出模型为 tar.gz（元数据 + 文件） |
| POST | `/api/v1/models/import-tar` | 从 tar.gz 上传导入模型 |

### 评测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/evaluations` | 创建评测 |
| GET | `/api/v1/evaluations` | 评测列表（按模型/版本/基准/状态过滤） |
| GET | `/api/v1/evaluations/benchmarks/compare` | 跨版本基准对比 |
| GET | `/api/v1/evaluations/{id}` | 获取评测详情 |
| PATCH | `/api/v1/evaluations/{id}` | 更新评测（状态/评分/指标） |
| DELETE | `/api/v1/evaluations/{id}` | 删除评测 |

### 同步

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/sync/versions/{id}/manifest` | 获取版本文件清单 |
| POST | `/api/v1/sync/push` | 推送模型到远端 FMH 实例 |
| POST | `/api/v1/sync/pull` | 从远端 FMH 实例拉取模型 |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/system/health` | 健康检查（含 MLX 状态） |
| GET | `/api/v1/system/storage` | 存储统计 |
| GET | `/api/v1/system/audit` | 查询审计日志 |
| GET | `/api/v1/system/export` | 导出全部数据（模型、租户、Webhook） |
| POST | `/api/v1/system/import` | 导入数据 |

### 认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/keys` | 创建 API Key（含角色） |
| GET | `/api/v1/auth/keys` | API Key 列表 |
| DELETE | `/api/v1/auth/keys/{id}` | 删除 API Key |
| POST | `/api/v1/auth/keys/{id}/deactivate` | 停用 API Key |

### 租户

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/tenants` | 创建租户 |
| GET | `/api/v1/tenants` | 租户列表 |
| GET | `/api/v1/tenants/{id}` | 获取租户 |
| PATCH | `/api/v1/tenants/{id}` | 更新租户 |
| DELETE | `/api/v1/tenants/{id}` | 删除租户 |

### Webhooks

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/webhooks` | 创建 Webhook |
| GET | `/api/v1/webhooks` | Webhook 列表 |
| GET | `/api/v1/webhooks/{id}` | 获取 Webhook |
| DELETE | `/api/v1/webhooks/{id}` | 删除 Webhook |

**Webhook 事件**（`events` 为逗号分隔列表，子串匹配）：

| 事件 | 触发时机 |
|------|----------|
| `model.created` | 注册新模型 |
| `model.deleted` | 删除模型 |
| `model.hot_reloaded` | FR-015 热重载切换服务版本 |
| `version.published` | 版本提升为 `published` |
| `version.deprecated` | 版本标记为 `deprecated` |
| `quantize.completed` | 量化任务完成 |
| `quantize.failed` | 量化任务失败 |
| `adapter.published` | #22 LoRA 适配器模型创建 |
| `adapter.merged` | #22 LoRA 合并任务完成（产出合并版本） |

### 部署

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/deployments` | 创建部署（自动加载模型至 MLX） |
| GET | `/api/v1/deployments` | 部署列表 |
| GET | `/api/v1/deployments/{id}` | 获取部署 |
| PATCH | `/api/v1/deployments/{id}` | 更新部署 |
| DELETE | `/api/v1/deployments/{id}` | 删除部署（自动从 MLX 卸载） |
| POST | `/api/v1/deployments/{id}/gray` | 启用灰度发布 |
| DELETE | `/api/v1/deployments/{id}/gray` | 关闭灰度发布 |
| POST | `/api/v1/deployments/{id}/scale` | 伸缩副本 |
| GET | `/api/v1/deployments/{id}/metrics` | 获取部署指标（MLX 状态 + 版本） |

### 推理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models/{id}/serve` | 加载模型至 Fusion-MLX |
| DELETE | `/api/v1/models/{id}/serve` | 卸载模型 |
| GET | `/api/v1/models/{id}/serve` | 获取服务状态 |
| POST | `/api/v1/models/{id}/hot-reload` | 零停机热重载到新版本（FR-015：预加载、切换服务记录、派发 `model.hot_reloaded`） |
| POST | `/api/v1/inference/{id}/chat` | 聊天补全（代理，灰度感知） |
| POST | `/api/v1/inference/{id}/completions` | 文本补全（代理） |
| POST | `/api/v1/inference/{id}/embeddings` | 嵌入（代理） |

### 量化

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/quantize` | 提交量化任务（2/4/6/8-bit） |
| GET | `/api/v1/quantize` | 量化任务列表 |
| GET | `/api/v1/quantize/running` | 当前运行中任务 |
| GET | `/api/v1/quantize/{task_id}` | 获取任务状态 |
| GET | `/api/v1/quantize/{task_id}/compare` | 对比源版本与量化版本指标 |
| POST | `/api/v1/quantize/layered` | 提交分层量化任务（按层 bits、group size、mode） |
| GET | `/api/v1/quantize/layered/jobs` | 分层量化任务列表 |
| GET | `/api/v1/quantize/layered/jobs/{job_id}` | 获取分层量化任务状态 |

### URL 下载

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models/{id}/versions/download-url` | 从 URL 下载版本（异步，SSRF 防护） |

### 集群

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/cluster/nodes` | 添加集群节点 |
| GET | `/api/v1/cluster/nodes` | 集群节点列表 |
| GET | `/api/v1/cluster/nodes/{id}` | 获取节点详情 |
| DELETE | `/api/v1/cluster/nodes/{id}` | 移除节点 |
| POST | `/api/v1/cluster/nodes/{id}/heartbeat` | 节点心跳 |

### 批量与同步

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models/sync` | 从远端 Hub 同步仓库 |
| POST | `/api/v1/models/batch/delete` | 批量删除模型 |
| POST | `/api/v1/models/batch/tag` | 批量打标签 |
| GET | `/api/v1/models/compare` | 对比模型（逗号分隔 ID） |

### 安全扫描

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/security/scan` | 触发模型/版本安全扫描 |
| GET | `/api/v1/security/scan/{scan_id}` | 按 ID 获取扫描结果 |
| GET | `/api/v1/security/scans` | 扫描列表（按 model_id、version_id、status 过滤） |

### 水印

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/watermark/embed` | 嵌入水印至模型/版本 |
| POST | `/api/v1/watermark/verify` | 验证水印签名 |
| GET | `/api/v1/watermark/list` | 水印列表（按 model_id、version_id 过滤） |

### 加密

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/encryption/encrypt` | 加密版本文件（AES-256 Fernet） |
| POST | `/api/v1/encryption/decrypt` | 解密版本文件 |
| GET | `/api/v1/encryption/status/{version_id}` | 查看版本加密状态 |

### 审批

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/approvals` | 提交审批请求（L1 自动，L2/L3 人工） |
| GET | `/api/v1/approvals` | 审批请求列表（按 model_id、status、level 过滤） |
| GET | `/api/v1/approvals/{req_id}` | 获取审批详情 |
| POST | `/api/v1/approvals/{req_id}/approve` | 批准 |
| POST | `/api/v1/approvals/{req_id}/reject` | 驳回 |

### Git LFS

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/gitlfs/objects/batch` | Git LFS v2 批量 API（上传/下载） |
| POST | `/api/v1/gitlfs/locks` | 创建模型路径锁 |
| GET | `/api/v1/gitlfs/locks` | 锁列表（按 model_id、path 过滤） |
| DELETE | `/api/v1/gitlfs/locks/{lock_id}` | 删除锁 |

### LoRA 合并

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/quantize/lora-merge` | 提交 LoRA 合并任务（基座 + 适配器，含量化） |
| GET | `/api/v1/quantize/lora-merge/{task_id}` | 获取 LoRA 合并任务状态 |

LoRA 模型使用 `model_type=lora` 并通过 `base_model_id` 外键指向基座 LLM。合并
运行器调用 Fusion-MLX 的 `POST {mlx_url}/v1/models/{name}/merge-adapter` 将适配器
融合为新的持久化 `ModelVersion`。在 Fusion-MLX 提供该端点之前（issue
fusion-mlx#584），任务会以明确的"升级 fusion-mlx"信息失败——Hub 侧对 404 容错且已就绪。

### 分布式任务

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/cluster/distributed-tasks` | 提交分布式任务（指定节点） |
| GET | `/api/v1/cluster/distributed-tasks/{task_id}` | 获取分布式任务状态 |

### 评分

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models/{id}/ratings` | 创建评分（1-5 分 + 可选评论） |
| GET | `/api/v1/models/{id}/ratings` | 评分列表（分页，含平均分） |
| GET | `/api/v1/models/{id}/ratings/summary` | 获取平均分 + 总数 |
| DELETE | `/api/v1/ratings/{rating_id}` | 删除评分 |

### 收藏

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models/{id}/favorites` | 收藏模型（重复返回 409） |
| GET | `/api/v1/models/{id}/favorites` | 模型收藏列表（分页） |
| GET | `/api/v1/favorites/me` | 当前用户收藏列表 |
| DELETE | `/api/v1/favorites/{favorite_id}` | 取消收藏 |

### 分支

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/models/{id}/branches` | 创建分支 |
| GET | `/api/v1/models/{id}/branches` | 分支列表（可选状态过滤） |
| GET | `/api/v1/branches/{branch_id}` | 获取分支详情 |
| PATCH | `/api/v1/branches/{branch_id}` | 更新分支 |
| DELETE | `/api/v1/branches/{branch_id}` | 删除分支 |
| POST | `/api/v1/branches/{branch_id}/merge` | 合并分支（状态设为 MERGED） |

### Prometheus 指标

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/metrics` | Prometheus 格式指标 |

### 硬件

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/hardware` | 获取硬件信息（芯片、VRAM、RAM、磁盘） |
| POST | `/api/v1/hardware/refresh` | 强制刷新硬件检测缓存 |

### 推荐

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/recommend` | 获取模型推荐（多维评分） |
| GET | `/api/v1/recommend/quick` | 快速推荐（Top 5） |

### 适配

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/adapt/assess` | 评估模型迁移等级（L0-L4），支持 source_format |
| POST | `/api/v1/adapt/plan` | 生成迁移方案（步骤 + 量化建议） |
| POST | `/api/v1/adapt/execute` | 执行完整适配流水线（assess→convert→quantize） |
| GET | `/api/v1/adapt/execute/{execution_id}` | 获取适配执行状态 |

### 基准数据

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/benchmarks` | 基准列表（按芯片、model_id、量化过滤） |
| GET | `/api/v1/benchmarks/{model_id}` | 获取模型最佳基准（按芯片、量化过滤） |

### 模型分析

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/analyze` | 分析模型结构（架构、层、参数、特殊算子） |

## SDK 客户端

`FusionModelHubClient` 提供覆盖全部 API 端点的同步 Python 客户端：

```python
from fusion_model_hub.sdk.client import FusionModelHubClient

client = FusionModelHubClient(base_url="http://localhost:11444", api_key="optional-key")

# 模型
client.create_model({"name": "qwen2.5-7b", "model_type": "llm"})
client.list_models(keyword="qwen")
client.get_model("model-id")
client.update_model("model-id", {"description": "Updated"})
client.delete_model("model-id")
client.import_from_hf({"hf_repo": "Qwen/Qwen2.5-7B"})

# 版本
client.promote_version("version-id")
client.benchmark_version("version-id")
client.rollback_version("version-id")
client.deprecate_version("version-id")

# 量化 & LoRA
client.start_quantize("source-version-id", quant_bits=4, calibration_dataset="my-dataset")
client.start_lora_merge("base-version-id", "lora-version-id")

# 安全 & 水印
client.start_security_scan("version-id")
client.embed_watermark("version-id", metadata='{"owner":"acme"}')
client.verify_watermark("version-id")

# 加密
client.encrypt_version("version-id")
client.decrypt_version("version-id")

# 审批
client.create_approval("version-id", level="L2", reason="Production release")
client.approve_request("req-id")
client.reject_request("req-id")

# Git LFS
client.gitlfs_batch("upload", [{"oid": "abc", "size": 1024}])
client.create_gitlfs_lock("models/qwen/safetensors")

# 集群
client.add_node("node-1", "http://node1:11444")
client.submit_distributed_task("inference", "version-id", target_node_ids=["node-1"])

# 评分
client.create_rating("model-id", score=5, comment="Excellent model")
client.list_ratings("model-id")
client.get_rating_summary("model-id")

# 收藏
client.add_favorite("model-id")
client.list_my_favorites()

# 分支
client.create_branch("model-id", name="experiment-v2")
client.list_branches("model-id")
client.merge_branch("branch-id")

# 硬件
client.get_hardware_info()
client.refresh_hardware()

# 推荐
client.recommend_models(task="llm", preference="speed", max_results=5)
client.quick_recommend(task="llm")

# 适配
client.assess_model("model-id", hf_repo="org/model", source_format="safetensors")
client.plan_migration("model-id", params_b=7.0, hf_repo="org/model")
client.execute_adaptation("model-id", quant_bits=4, params_b=7.0)
client.get_adapt_execution("execution-id")

# 基准
client.list_benchmarks(chip="M4 Pro", model_id="qwen2.5-7b")
client.get_benchmark("qwen2.5-7b", chip="M4 Pro", quant="4bit")

# 分析
client.analyze_model(model_path="/path/to/model", hf_repo="org/model")

# 分层量化
client.start_layered_quantize("model-id", default_bits=4, layer_rules=[{"pattern": ".*lm_head", "bits": 8}])
client.get_layered_quantize_job("job-id")
client.list_layered_quantize_jobs()
```

### 异步 SDK 客户端

```python
from fusion_model_hub.sdk.async_client import AsyncFusionModelHubClient

async with AsyncFusionModelHubClient(base_url="http://localhost:11444") as client:
    models = await client.list_models()
    await client.create_rating("model-id", score=5)
```

## 使用示例

```bash
# 创建模型
curl -X POST http://localhost:11444/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{"name": "qwen2.5-7b", "model_type": "llm", "architecture": "qwen2", "params_size": "7B"}'

# 上传版本（含文件）
curl -X POST http://localhost:11444/api/v1/models/{model_id}/versions \
  -F "version=1.0.0" \
  -F "format=mlx" \
  -F "quantization=4bit" \
  -F "file=@model_weights.bin"

# 从 HuggingFace 导入（仅元数据）
curl -X POST http://localhost:11444/api/v1/models/import/hf \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-7B"}'

# 从 HuggingFace 导入（含下载）
curl -X POST http://localhost:11444/api/v1/models/import/hf \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-7B", "download": true}'

# 搜索模型
curl "http://localhost:11444/api/v1/models/search?keyword=qwen&quantization=4bit&sort_by=benchmark_score"

# 获取模型推荐
curl "http://localhost:11444/api/v1/models/recommend?task_type=text-generation&max_params=7B&limit=5"

# 提交量化任务
curl -X POST http://localhost:11444/api/v1/quantize \
  -H "Content-Type: application/json" \
  -d '{"source_version_id": "<version_id>", "quant_bits": 4}'

# 对比量化前后
curl "http://localhost:11444/api/v1/quantize/{task_id}/compare"

# 从 URL 下载版本
curl -X POST http://localhost:11444/api/v1/models/{model_id}/versions/download-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://hf-mirror.com/...", "version": "1.0.0-4bit"}'

# 导出模型为 tar.gz
curl -o model.tar.gz "http://localhost:11444/api/v1/models/{model_id}/export"

# 创建评测
curl -X POST http://localhost:11444/api/v1/evaluations \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "benchmark_name": "mmlu", "status": "running"}'

# 推送模型到远端 FMH 实例
curl -X POST http://localhost:11444/api/v1/sync/push \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "target_url": "https://other-fmh.example.com"}'

# 晋升版本生命周期（DRAFT→TESTING→PUBLISHED）
curl -X POST http://localhost:11444/api/v1/versions/{version_id}/promote

# 安全扫描
curl -X POST http://localhost:11444/api/v1/security/scan \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "scan_type": "full"}'

# 嵌入水印
curl -X POST http://localhost:11444/api/v1/watermark/embed \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "metadata": "{\"owner\": \"acme\"}"}'

# 加密版本文件
curl -X POST http://localhost:11444/api/v1/encryption/encrypt \
  -H "Content-Type: application/json" \
  -d '{"version_id": "..."}'

# 提交审批请求
curl -X POST http://localhost:11444/api/v1/approvals \
  -H "Content-Type: application/json" \
  -d '{"version_id": "...", "level": "L2", "reason": "Production release"}'

# LoRA 合并
curl -X POST http://localhost:11444/api/v1/quantize/lora-merge \
  -H "Content-Type: application/json" \
  -d '{"base_version_id": "...", "lora_version_id": "...", "quant_bits": 4}'

# 提交分布式任务
curl -X POST http://localhost:11444/api/v1/cluster/distributed-tasks \
  -H "Content-Type: application/json" \
  -d '{"model_id": "...", "version_id": "...", "target_nodes": ["node-1", "node-2"]}'

# 评估模型适配等级
curl -X POST http://localhost:11444/api/v1/adapt/assess \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3.2-1b", "hf_repo": "meta-llama/Llama-3.2-1B", "source_format": "safetensors"}'

# 执行完整适配流水线
curl -X POST http://localhost:11444/api/v1/adapt/execute \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3.2-1b", "quant_bits": 4, "params_b": 1.0}'

# 获取适配执行状态
curl http://localhost:11444/api/v1/adapt/execute/{execution_id}

# 基准列表
curl "http://localhost:11444/api/v1/benchmarks?chip=M4+Pro&model_id=qwen2.5-7b"

# 获取模型最佳基准
curl "http://localhost:11444/api/v1/benchmarks/qwen2.5-7b?chip=M4+Pro&quant=4bit"

# 分析模型结构
curl -X POST http://localhost:11444/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-7B"}'

# 提交分层量化（按层 bits）
curl -X POST http://localhost:11444/api/v1/quantize/layered \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-7b", "default_bits": 4, "layer_rules": [{"pattern": ".*lm_head", "bits": 8}], "quant_group_size": 64}'

# 获取分层量化任务状态
curl http://localhost:11444/api/v1/quantize/layered/jobs/{job_id}
```

## 架构

```
fusion_model_hub/
├── db/
│   ├── models.py          # SQLAlchemy ORM
│   ├── database.py        # 异步引擎 & 会话工厂（aiosqlite）
│   └── crud.py            # 异步 CRUD 操作，字段白名单
├── storage/
│   ├── base.py            # StorageBackend 抽象基类
│   ├── local_store.py     # LocalStore: 分块上传、SHA256、组装、加密
│   └── minio_store.py     # MinioStore: S3 兼容对象存储
├── server/
│   ├── app.py             # FastAPI 应用工厂（lifespan + 备份调度）
│   ├── config.py          # Settings 数据类（环境变量 + MinIO + 备份 + TLS 配置）
│   ├── deps.py            # 依赖注入（Session, Store[Backend], Settings）
│   ├── auth.py            # 认证中间件（RBAC + 所有者强制）
│   ├── tasks.py           # 异步任务管理器（量化任务含校准）
│   ├── backup.py          # 自动备份调度（可配置间隔，JSON 导出）
│   ├── metrics.py         # Prometheus 指标中间件 + /metrics 端点
│   ├── __main__.py        # CLI 入口（serve/export/import/migrate + TLS）
│   └── routers/
│       ├── models.py      # /api/v1/models + HF 导入 + 同步/批量/对比/搜索/推荐
│       ├── versions.py    # /api/v1/versions + 生命周期 + 晋升 + 基准 + 指标 + tar 导出/导入
│       ├── quantize.py    # /api/v1/quantize + 对比 + LoRA 合并
│       ├── inference.py   # /api/v1/inference 代理 + 灰度路由
│       ├── auth.py        # /api/v1/auth Key 管理 + RBAC 角色
│       ├── cluster.py     # /api/v1/cluster 节点 + 心跳 + 分布式任务
│       ├── system.py      # /api/v1/system（健康 + MLX + 审计 + 导出/导入）
│       ├── tenants.py     # /api/v1/tenants CRUD
│       ├── webhooks.py    # /api/v1/webhooks + 事件派发 + 重试
│       ├── deployments.py # /api/v1/deployments + 灰度 + 伸缩 + 指标 + MLX 集成
│       ├── evaluations.py # /api/v1/evaluations + 基准对比
│       ├── sync.py        # /api/v1/sync（推送/拉取/清单）
│       ├── security.py    # /api/v1/security 扫描
│       ├── watermark.py   # /api/v1/watermark 嵌入/验证
│       ├── encryption.py  # /api/v1/encryption 加密/解密/状态
│       ├── approvals.py   # /api/v1/approvals 提交/批准/驳回
│       ├── gitlfs.py      # /api/v1/gitlfs 批量 + 锁
│       ├── ratings.py     # /api/v1/models/{id}/ratings CRUD + 汇总
│       ├── favorites.py   # /api/v1/models/{id}/favorites + /me
│       ├── branches.py    # /api/v1/models/{id}/branches + 合并
│       ├── hardware.py    # /api/v1/hardware（代理 MLX + 刷新）
│       ├── recommend.py   # /api/v1/recommend（多维评分 + 批量 MLX）
│       ├── adapt.py       # /api/v1/adapt（评估 + 方案 + 执行流水线）
│       ├── benchmarks.py  # /api/v1/benchmarks（代理 MLX 基准数据）
│       └── analyze.py     # /api/v1/analyze（代理 MLX 模型结构分析）
├── sdk/
│   ├── client.py          # FusionModelHubClient — 同步 Python SDK
│   ├── async_client.py    # AsyncFusionModelHubClient — 异步 Python SDK
│   └── models.py          # Pydantic 请求/响应模型
├── api/
│   └── base_binding.py    # FusionMLX HTTP 客户端
├── hardware/
│   ├── __init__.py        # 导出 HardwareDetector, HardwareProfile
│   ├── types.py           # ChipGeneration, GPUProfile, CPUProfile, HardwareProfile
│   └── detector.py        # HardwareDetector — MLX 硬件检测，5 分钟缓存
├── recommend/
│   ├── __init__.py        # 导出 RecommendEngine, ModelRecommendation
│   ├── types.py           # RecommendRequest, ModelRecommendation, RecommendResponse
│   ├── scorer.py          # 多维评分（硬件适配、质量、速度、热度）
│   └── engine.py          # RecommendEngine — 批量 MLX 推荐 + 评分器降级
├── adapt/
│   ├── __init__.py        # 导出 AdaptDecisionEngine, AdaptationLevel
│   ├── types.py           # AdaptationLevel（L0-L4）, MigrationPlan, AdaptationResult
│   ├── migration.py       # 迁移方案生成 + 量化建议
│   └── decision.py        # AdaptDecisionEngine — MLX migration-level + analyze 增强 + 本地降级
├── cache/
│   ├── __init__.py        # 导出 CacheManager, CacheLevel
│   ├── types.py           # CacheLevel, CacheEntry, CacheStats
│   └── manager.py         # 三级缓存（raw/converted/quantized）+ GC + 校验 + MLX 版本感知
├── cli/
│   ├── __init__.py        # 导出 typer app
│   ├── main.py            # fmh CLI 入口（hardware, version, 子应用）
│   ├── download.py        # fmh download（hf, url）
│   ├── recommend.py       # fmh recommend（models, quick）
│   ├── list_cmd.py        # fmh list（local, remote, stats）
│   └── analyze.py         # fmh analyze（assess, plan）
├── convert/
│   └── converter.py       # 通过 Fusion-MLX 模型转换
├── manage/
│   └── manager.py         # 本地模型管理器
└── repo/
    ├── models.py           # 数据模型（ModelInfo）
    ├── registry.py         # 内存模型目录
    └── downloader.py       # 异步下载，支持断点续传
```

## 配置

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FMH_DATA_DIR` | `./data` | 数据目录（DB 和文件） |
| `FMH_MLX_URL` | `http://localhost:11434` | Fusion-MLX 服务 URL |
| `FMH_AUTH_ENABLED` | `false` | 启用 API Key 认证 |
| `FMH_CORS_ORIGINS` | `*` | 允许的 CORS 来源 |
| `FMH_MAX_UPLOAD_SIZE_MB` | `500` | 最大上传文件大小 |
| `FMH_DB_URL` | `sqlite+aiosqlite:///data/fmh.db` | 数据库 URL（支持 PostgreSQL） |
| `FMH_ALEMBIC_URL` | `sqlite://` | Alembic 迁移用同步 DB URL |
| `FMH_STORAGE_TYPE` | `local` | 存储后端类型（`local` 或 `minio`） |
| `FMH_MINIO_ENDPOINT` | `` | MinIO 端点（storage_type=minio 时） |
| `FMH_MINIO_ACCESS_KEY` | `` | MinIO Access Key |
| `FMH_MINIO_SECRET_KEY` | `` | MinIO Secret Key |
| `FMH_MINIO_BUCKET` | `fusion-models` | MinIO 桶名 |
| `FMH_MINIO_SECURE` | `true` | MinIO 使用 HTTPS |
| `FMH_BACKUP_DIR` | `` | 自动备份 JSON 文件目录 |
| `FMH_TLS_CERTFILE` | `` | TLS 证书文件路径 |
| `FMH_TLS_KEYFILE` | `` | TLS 私钥文件路径 |

CLI 选项覆盖环境变量：`--host`、`--port`、`--data-dir`、`--db-url`、`--mlx-url`、`--log-level`、`--tls-certfile`、`--tls-keyfile`

## CLI

`fmh` 命令提供基于 typer 的 CLI：

```bash
# 查看硬件信息
fmh hardware

# 从 HuggingFace 镜像下载模型
fmh download hf mlx-community/Llama-3.2-1B-Instruct-4bit --mirror https://hf-mirror.com

# 从 URL 下载
fmh download url https://example.com/model.mlx my-model

# 获取模型推荐
fmh recommend models --task llm --preference speed --max-results 5
fmh recommend quick --task llm

# 列出本地模型
fmh list local
fmh list remote --limit 20
fmh list stats

# 分析模型适配
fmh analyze assess mlx-community/Llama-3.2-1B-Instruct-4bit
fmh analyze plan mlx-community/Llama-3.2-1B-Instruct-4bit

# 版本
fmh version
```

## 开发

```bash
source .venv/bin/activate
pip install -e ".[test]"

# 运行全部测试
pytest

# 仅运行 API 集成测试
pytest tests/test_api.py -v

# 运行覆盖率
pytest --cov=fusion_model_hub --cov-report=term-missing

# 启动 Fusion-MLX（用于集成测试）
~/claude-home/fusion-mlx/start.sh start
```

## 模型下载镜像

使用 `https://hf-mirror.com` 在 HuggingFace 访问受限区域下载模型。
