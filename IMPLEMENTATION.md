# Fusion Model Hub — 三步实施计划

User instruction: "第一步，落地实施plan.md中遗留的几项，第二步，prd文档中提到的企业级的需求全部落地，第三步，全量补齐测试用例，用例覆盖率90%+"

This file is a plan document only — no importers/callers. It references:
- `plan.md` (existing phases doc), `ar.md` (PRD doc)
- API routers in `fusion_model_hub/server/routers/`
- ORM models in `fusion_model_hub/db/models.py`
- CRUD in `fusion_model_hub/db/crud.py`

## 当前状态

- 总测试：117个，覆盖率 63.9% (1890/2956行)
- plan.md Phase 2-6 大部分已完成，遗留2项
- ar.md 企业级功能大部分未实现

---

## 第一步：plan.md 遗留项 (2项)

### 1.1 版本晋升端点 `POST /api/v1/versions/{id}/promote`

- 在 `server/routers/versions.py` 新增 `promote_version` 端点
- 按标准流程自动晋升：DRAFT→TESTING→PUBLISHED，每步校验 VALID_TRANSITIONS
- 逻辑：从当前状态开始，依次调用 `crud.update_version_status`，直到 PUBLISHED
- 如果某步失败（非法转换），返回当前已到达的状态和错误信息
- 测试覆盖：DRAFT→PUBLISHED 全流程、中间状态晋升、非法状态晋升

### 1.2 选择性导出 `--models` 参数

- CLI `export --models id1,id2` 过滤指定模型
- API `GET /api/v1/system/export?models=id1,id2` 端点支持过滤
- 修改 `_run_export()` 和 `__main__.py` 增加 models 参数
- 新增 `server/routers/system.py` 的 export/import 端点（plan.md 6.1）

---

## 第二步：ar.md 企业级需求 (8项)

### 2.1 安全扫描 (FR-025)

**新增**: `server/routers/security.py`, `db/models.py` 增加 SecurityScan 表
**端点**: POST /security/scan, GET /security/scan/{id}, GET /security/scans
**实现**: 扫描模型配置恶意payload、不安全依赖、硬编码密钥

### 2.2 模型水印与溯源 (NFR-003)

**新增**: `server/routers/watermark.py`, `db/models.py` 增加 Watermark 表
**端点**: POST /watermark/embed, GET /watermark/verify, GET /watermark/list
**实现**: 基于元数据水印（owner+timestamp+hash签名），权重水印为扩展点

### 2.3 静态加密 (NFR-002)

**新增**: `server/routers/encryption.py`, `ModelVersion` 增加 encrypted 字段
**端点**: POST /encryption/encrypt, POST /encryption/decrypt, GET /encryption/status/{version_id}
**实现**: AES-256-GCM, 密钥由环境变量 FMH_ENCRYPTION_KEY 管理

### 2.4 审批工作流 (9.7 L1/L2/L3)

**新增**: `server/routers/approvals.py`, `db/models.py` 增加 ApprovalRequest 表
**端点**: POST /approvals, GET /approvals, POST /approvals/{id}/approve, POST /approvals/{id}/reject
**级别**: L1自动审批, L2单人审批, L3三级审批

### 2.5 Git LFS 协议支持 (FR-027)

**新增**: `server/routers/gitlfs.py`
**端点**: POST /gitlfs/objects/batch, POST /gitlfs/locks, GET /gitlfs/locks
**实现**: Git LFS v2 batch API，对象存储委托 StorageBackend

### 2.6 SDK (FR-026)

**新增**: `sdk/client.py`, `sdk/models.py`, `sdk/async_client.py`
**类**: FusionModelHubClient 封装所有 API 调用

### 2.7 LoRA 合并量化 (FR-009)

**扩展**: `server/routers/quantize.py`, 新增 LoraMergeTask 表
**端点**: POST /quantize/lora-merge, GET /quantize/lora-merge/{task_id}
**实现**: 委托 Fusion-MLX HTTP API

### 2.8 分布式多节点 (FR-014)

**扩展**: `server/routers/cluster.py`, 新增 DistributedTask 表
**端点**: POST /cluster/distribute, GET /cluster/distribute/{task_id}, POST /cluster/nodes/{id}/sync

---

## 第三步：全量测试补齐 (目标 90%+)

当前 63.9%，需补齐：

### 高优先（<30%）
- `__main__.py` (0%), `minio_store.py` (0%), `sync.py` (26%), `tasks.py` (27%), `inference.py` (27%)

### 中优先（30-70%）
- `evaluations.py` (42%), `auth.py` (45%), `versions.py` (57%), `quantize.py` (57%), `app.py` (63%), `webhooks.py` (67%), `models.py` (67%), `deployments.py` (69%), `crud.py` (74%), `local_store.py` (75%)

### 新增功能同步测试

---

## 实施顺序

Step 1.1 promote → Step 1.2 选择性导出 → Step 2.1~2.8 企业级功能 → Step 3 测试补齐 → 最终验证 ≥ 90%
