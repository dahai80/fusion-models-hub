# 剩余 Phase 实施计划

用户指令："把所有剩余未实现的按照你给出的优先级落地实施"

涉及文件导入/调用关系:
- db/models.py → 被 db/crud.py, server/routers/*.py, server/auth.py, server/tasks.py 引用
- db/crud.py → 被 server/routers/*.py, server/auth.py, server/tasks.py 调用
- server/auth.py → 被 server/app.py 注册为中间件, 被 server/routers/auth.py 间接使用
- 新增 routers → 被 server/app.py include_router 注册

受影响 API: 所有 /api/v1/* 端点（RBAC 校验影响全部写操作），新增 /api/v1/tenants, /api/v1/webhooks, /api/v1/deployments, /api/v1/sync, /api/v1/models/recommend, /api/v1/system/export|import, /api/v1/versions/{id}/manifest

数据 schema 变更: 新增 Tenant, Webhook, Deployment 三张表; ApiKey/Model/AuditLog 新增字段

---

## 优先级 1: RBAC 角色权限控制 (Phase 4.2)

### 现状
- `ApiKey.permissions` 字段为 `"read,write"` 字符串，中间件不校验具体角色
- `auth_middleware` 只验证 key 是否有效，不区分 admin/developer/viewer

### 实施
1. **db/models.py** — `ApiKey` 表新增 `role` 字段: `admin/developer/viewer`，默认 `developer`
2. **db/crud.py** — `create_api_key` 接受 `role` 参数
3. **server/auth.py** — `auth_middleware` 增加角色校验:
   - admin: 全部操作
   - developer: GET + POST/PUT，不可 DELETE
   - viewer: 仅 GET
4. **server/routers/auth.py** — `create_api_key` 接受 `role` 参数
5. **tests/test_api.py** — 新增 RBAC 测试

---

## 优先级 2: 多租户隔离 (Phase 4.4)

### 实施
1. **db/models.py** — 新增 `Tenant` ORM 表; Model/ApiKey/AuditLog 新增 `tenant_id`
2. **db/crud.py** — 新增 tenant CRUD; list_* 支持 tenant_id 过滤
3. **server/routers/tenants.py** — POST/GET/DELETE /api/v1/tenants
4. **server/app.py** — 注册 tenants router
5. **server/auth.py** — 从 ApiKey.tenant_id 注入 request.state.tenant_id
6. **tests/test_api.py** — 新增多租户测试

---

## 优先级 3: Webhook 事件通知 (Phase 4.5)

### 实施
1. **db/models.py** — 新增 `Webhook` ORM 表
2. **db/crud.py** — webhook CRUD
3. **server/routers/webhooks.py** — POST/GET/DELETE /api/v1/webhooks
4. **server/webhooks.py** — 事件分发器: dispatch_event, HMAC 签名, 重试 3 次
5. **集成点** — models/versions/tasks 关键操作后调用 dispatch_event
6. **tests/test_api.py** — 新增 webhook 测试

---

## 优先级 4: Deployment 部署模型 (Phase 5.1)

### 实施
1. **db/models.py** — 新增 `Deployment` ORM 表
2. **db/crud.py** — deployment CRUD
3. **server/routers/deployments.py** — POST/GET/DELETE /api/v1/deployments
4. **inference.py** — serve/unload 同步写 Deployment 记录
5. **tests/test_api.py** — 新增部署测试

---

## 优先级 5: 扩缩容 + 灰度发布 (Phase 5.2-5.3)

### 实施
1. **db/models.py** — Deployment 新增 gray_target_version_id, gray_traffic_ratio
2. **server/routers/deployments.py** — PUT /{id}/scale, PUT /{id}/gray
3. **inference.py** — 灰度路由逻辑
4. **tests/test_api.py** — 新增灰度测试

---

## 优先级 6: Phase 6 分布式 & 企业级

### 6.1 离线导入导出
- POST /api/v1/system/export, POST /api/v1/system/import

### 6.2 差分同步引擎
- GET /api/v1/versions/{id}/manifest, POST /api/v1/sync/push, POST /api/v1/sync/pull

### 6.3 智能模型推荐
- GET /api/v1/models/recommend

### 6.4 存储后端抽象
- storage/backend.py 抽象接口, storage/minio_store.py MinIO 实现

### 6.5 DB 迁移路径
- alembic/ 初始化 + 初始迁移

### 6.6 CLI 增强
- __main__.py export/import/migrate 子命令
