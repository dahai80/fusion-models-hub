# Fusion-Model-Hub 全面严格审计报告

> **审计对象**：`fusion-model-hub` — Fusion-MLX 生态的统一模型仓库与管理中心
> **审计范围**：系统架构、安全、可靠性、可扩展性、代码质量、内存泄漏风险、系统完整性
> **审计方式**：纯只读代码审计（不修改任何代码）
> **审计日期**：2026-07-26
> **审计员**：AtomCode (GLM-5.2)
> **代码基线**：`main` 分支 `bf22794` + 未提交工作区改动

---

## 一、综合评分总览

| 维度 | 评分（/10） | 等级 | 一句话结论 |
|------|:----------:|:----:|-----------|
| 系统架构 | **7.0** | B | 分层清晰、依赖注入规范，但缺统一网络出口与配置分层 |
| 安全 | **5.5** | C+ | SSRF/认证基础具备，但无速率限制、无统一安全模块、审计日志有缺口 |
| 可靠性 | **5.0** | C | 缺重试/退避、缺跨语句事务、`download_version` 有竞态、孤儿任务清理时机错误 |
| 可扩展性 | **5.5** | C+ | 全异步 + 分页做得好，但无连接池配置、无缓存、有 N+1 与全表扫描 |
| 代码质量与架构 | **6.5** | B- | 类型标注齐全、字段白名单规范，但 SSRF 逻辑三处重复、多处 `dict` 弱类型 |
| 内存泄漏风险 | **5.5** | C+ | `_loaded_models` 无锁、`asyncio.create_task` 未追踪、chunk 临时文件无清理 |
| 系统完整性 | **6.0** | C+ | 外键+状态机设计好，但无优雅停机、无文件-DB 一致性、`_reconcile` 在 `init_deps` 前调用会崩 |
| **加权综合** | **5.9 / 10** | **C+** | **可用但未达生产级；安全与可靠性是最大短板** |

**权重说明**：安全 0.20、可靠性 0.20、系统完整性 0.15、内存泄漏 0.15、可扩展性 0.10、架构 0.10、代码质量 0.10。

**风险等级图例**：🔴 高（阻断上线/数据损坏）、🟠 中（影响稳定性/可运维性）、🟡 低（代码异味/可延后）。

---

## 二、系统架构审计（7.0 / 10，B）

### 2.1 架构优点

1. **分层清晰**：`db/`（数据层）→ `server/`（API 层）→ `repo/convert/manage/`（业务层）→ `api/`（外部绑定），职责边界明确。
2. **依赖注入规范**：`deps.py` 用 `Annotated[...]` + `Depends` 暴露 `SessionDep/StoreDep/SettingsDep`，符合 FastAPI 最佳实践。
3. **零直接 ML 依赖**：严格遵守 CLAUDE.md 约束，所有推理/转换/校验 100% 走 `http://localhost:11434`，不直接 import `mlx/torch/transformers`。
4. **ASGI 生命周期正确**：`app.py` 用 `@asynccontextmanager lifespan`（非已废弃的 `on_event`），测试需手动 `init_deps()`（CLAUDE.md 已说明）。
5. **字段白名单防越权**：`crud.py` 的 `_MODEL_UPDATABLE` / `_VERSION_UPDATABLE` / `_TASK_UPDATABLE` 显式列出可更新字段，避免 `hasattr` 滥用导致的越权写入。

### 2.2 架构缺陷

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| ARCH-1 | 🟠 | 全局 | **无统一网络出口**。`httpx.AsyncClient` 在 9+ 处分散实例化（`inference.py` 6 处、`models.py`、`versions.py`、`system.py`、`base_binding.py`、`converter.py`、`downloader.py`），各自设 `timeout`，无统一 `User-Agent`/`verify`/重定向上限 | 安全策略难统一；TLS 默认开但无显式声明；重定向攻击面不可控 |
| ARCH-2 | 🟠 | `config.py` | **Settings 是扁平 dataclass**。`db_url` 在 `__post_init__` 里依赖 `data_dir`，但 `mlx_url` 的 env 回退只在 `__post_init__` 里对空串生效，逻辑分散 | 配置来源（env/文件/默认值）无单一真相，易错配 |
| ARCH-3 | 🟠 | `deps.py` | **模块级单例 `_settings/_session_factory/_store` 用全局变量**。`init_deps()` 可被多次调用覆盖，无线程安全 | 多 worker / 测试并行时易串状态 |
| ARCH-4 | 🟡 | `server/routers/` | **路由文件同时承担序列化、校验、CRUD 调用**。`_model_to_dict` / `_version_to_dict` 等序列化函数散落各 router，未抽到独立 `schemas/` 层 | 重复序列化逻辑；难做 OpenAPI schema 统一 |
| ARCH-5 | 🟡 | `repo/registry.py` | **`ModelRegistry._models` 是类级 dict**，注释说"in-memory catalog"，但无任何 router 接入它，疑似死代码 | 混淆"内存注册表"与"DB 注册表"的边界 |
| ARCH-6 | 🟡 | `server/app.py:73-82` | **请求日志中间件在异常路径下不记录耗时**，且对 `/api/v1/system/health` 这类探针无降采样 | 日志噪声；高频探针打满日志 |

### 2.3 架构评分明细

| 子项 | 得分 | 说明 |
|------|:----:|------|
| 分层与职责边界 | 8/10 | 分层好，但 router 承担过多 |
| 依赖注入 | 8/10 | 规范，但全局单例不线程安全 |
| 配置管理 | 6/10 | 扁平 dataclass，env 回退逻辑分散 |
| 网络出口统一性 | 4/10 | 9+ 处分散 `AsyncClient`，无统一安全策略 |
| 路由组织 | 7/10 | 静态路径在动态路径前，正确；但序列化未抽离 |
| **小计** | **7.0** | |

---

## 三、安全审计（5.5 / 10，C+）

### 3.1 认证与授权

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| SEC-1 | 🔴 | `auth.py:34-47` | **认证默认关闭**（`auth_enabled=False`）。生产若忘开 `FMH_AUTH_ENABLED`，所有写接口裸奔 | 任意客户端可创建/删除模型与版本 |
| SEC-2 | 🟠 | `auth.py` | **无速率限制**。全代码库 grep `Limiter\|slowapi\|rate.?limit` 零命中。API key 暴力破解、推理刷量、下载刷量均无防护 | 暴力破解 / DoS |
| SEC-3 | 🟠 | `auth.py:42-53` | **API key 比较非常数时间**。`verify_api_key` 内部用 `==` 比 `key_hash`（SHA256 hex），虽然 hash 后比较在一定程度上缓解，但 `crud.py:359-368` 的 `verify_api_key` 对"key 不存在"和"key 存在"路径耗时不同 | 时序侧信道泄露 key 存在性 |
| SEC-4 | 🟠 | `auth.py` | **审计日志只记录写操作**，读操作（下载模型文件、列出版本、推理调用）无审计。下载模型文件属于敏感数据外流 | 合规审计盲区 |
| SEC-5 | 🟡 | `auth.py:12-21` | **`PUBLIC_PATHS` 用 `startswith` 前缀匹配**。`/api/v1/auth/keys` 被放行，意味着 `DELETE /api/v1/auth/keys/{id}` 也无需认证 | 任意客户端可删 API key |
| SEC-6 | 🟡 | `db/models.py:151-168` | **API key 仅 SHA256 单层 hash**，无 salt、无 PBKDF2/argon2。若 DB 泄露，彩虹表/暴力破解易还原 key | DB 泄露后 key 可逆 |
| SEC-7 | 🟡 | `auth.py` | **无 RBAC**。`ApiKey.permissions` 字段存了 `"read,write"` 字符串但从未被检查；所有有效 key 权限等价 | 权限粒度不足 |

### 3.2 SSRF 与网络出口

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| SEC-8 | 🔴 | `repo/downloader.py:52` | **下载用 `follow_redirects=True` 但 SSRF 校验只在入口做一次**。`download_version_from_url` 调 `_validate_download_url` 后，`ModelDownloader.download` 跟随重定向到任意主机（包括 `169.254.169.254` 元数据端点） | SSRF 绕过 → 云元数据泄露 |
| SEC-9 | 🟠 | `models.py:138-148` & `versions.py:272-285` | **SSRF 黑名单逻辑三处重复且不一致**。`_is_internal_hostname`（models.py）不处理 `::ffff:` 映射 IPv6；`_validate_download_url`（versions.py）处理了但漏了 `fd00:ec2::254` 等云元数据 IPv6 | 维护漂移 → 漏洞 |
| SEC-10 | 🟠 | `versions.py:272-285` | **SSRF 校验不解析 IP 字面量**。只做字符串前缀匹配（`startswith("10.")`、`startswith("fc")`），`0xA.0.0.1`（十六进制）、`http://2130706433/`（十进制）、`http://[::ffff:127.0.0.1]` 等编码可绕过 | SSRF 绕过 |
| SEC-11 | 🟠 | 全局 `AsyncClient` | **无统一 `verify=True` 显式声明、无 `max_redirects` 上限**。默认值虽安全，但无显式固化 | 配置漂移风险 |
| SEC-12 | 🟡 | `models.py:160-201` `sync_registry` | **同步接口把 `body.source_url` 拼成 `{url}/api/v1/models`**，若 `source_url` 含 `?` 或 `#` 会污染请求；且未限制响应体大小 | 远程 OOM / 请求分裂 |

### 3.3 输入校验

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| SEC-13 | 🟠 | `inference.py:117,136,155` | **三个推理端点用 `body: dict` 裸接收**，无 Pydantic 模型校验。`max_tokens`/`temperature` 无上限，可传 `max_tokens=10_000_000` 打爆 MLX | 输入注入 / 资源耗尽 |
| SEC-14 | 🟠 | `models.py:230` `import_from_hf` | **同样 `body: dict`**，`hf_repo` 无 `min_length`/`max_length`/格式校验，可传 `../../etc/passwd` 之类 | 路径注入到 HF API |
| SEC-15 | 🟡 | `models.py:50-52` `HfImportRequest` | `hf_repo` 有 `min_length=1` 但无 `max_length`，可传超长串打满日志 | 日志注入 |
| SEC-16 | 🟡 | `versions.py:64-106` `upload_version` | **单文件上传无总大小上限**（只对非分块上传限 100MB），分块上传 `total_chunks` 上限 10000 × 20MB = 200GB，但无总体积校验 | 磁盘耗尽攻击 |

### 3.4 安全评分明细

| 子项 | 得分 | 说明 |
|------|:----:|------|------|
| 认证 | 5/10 | 默认关闭、无 RBAC、无速率限制 |
| 授权 | 4/10 | permissions 字段形同虚设 |
| SSRF 防护 | 4/10 | 有基础但重定向绕过 + 编码绕过 |
| 输入校验 | 5/10 | 推理/HF import 用裸 dict |
| 审计日志 | 5/10 | 只记写操作、下载无审计 |
| 密钥存储 | 6/10 | SHA256 无 salt |
| **小计** | **5.5** | |

---

## 四、可靠性审计（5.0 / 10，C）

### 4.1 错误处理与重试

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| REL-1 | 🔴 | 全局 | **零重试/退避机制**。grep `retry\|backoff\|attempts\|max_retries` 零命中。MLX 服务重启、DNS 抖动、5xx 瞬时错误全部直接冒泡为用户可见 500 | 瞬时故障 = 用户报错 |
| REL-2 | 🟠 | `tasks.py:42-128` `_run_quantize` | **`ModelConverter.quantize` 失败时任务卡在 `RUNNING`**。`_run_quantize` 的 `except` 分支会写 `FAILED`，但若 `converter.quantize` 内部 `httpx` 调用挂起（600s timeout），任务在整个超时期间一直 `RUNNING`，且 `_reconcile_orphaned_tasks` 只在启动时跑一次 | 任务永久卡死 |
| REL-3 | 🟠 | `app.py:84-87` `global_exception_handler` | **全局异常处理器吃掉所有异常返回 500**，不区分 `InvalidTransition`（应 409）、`SecurityError`（应 400）。虽然 router 内部已 try/except 转换，但任何漏网的领域异常都会变成无信息 500 | 错误信息丢失 |
| REL-4 | 🟡 | `inference.py:72-75` 等 | **`httpx.HTTPStatusError` 的 `e.response.text` 直接回传给客户端**。MLX 返回的内部错误信息（可能含路径、栈）泄露给调用方 | 信息泄露 |

### 4.2 事务安全

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| REL-5 | 🔴 | `versions.py:189-208` `download_version` | **竞态条件 + 计数泄漏**。流程：`get_version` → `get_file` → `increment_download` → `FileResponse`。若 `FileResponse` 在流式传输中失败（磁盘读错、连接断），`download_count` 已 +1 但用户没拿到文件；反之若先发 `FileResponse` 再 `increment_download`，崩溃会漏计。当前实现是"先计数再发文件"，错误方向 | 下载数失真 |
| REL-6 | 🟠 | `models.py:323-337` `update_model` | **跨语句无事务包裹**。`update_model` 先 `crud.update_model`（内部 commit），再 `crud.set_tags`（内部 commit），再 `session.refresh(m)`。若 `set_tags` 失败，model 已更新但 tags 未变，部分提交 | 数据半一致 |
| REL-7 | 🟠 | `tasks.py:86-97` | **创建输出 version 与更新任务状态分两次 commit**。若 `create_version` 成功但后续 `update_quantize_task(COMPLETED)` 失败，DB 里会多出一个"孤儿 version"，且任务仍显示 `RUNNING` | 孤儿数据 |
| REL-8 | 🟡 | `crud.py` 全局 | **每个 CRUD 函数内部 `commit()`**，调用方无法把多个 CRUD 组成一个事务。无 `transactional()` 上下文管理器 | 难做原子多步 |

### 4.3 竞态条件

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| REL-9 | 🟠 | `inference.py:15` `_loaded_models` | **`_loaded_models` 无锁**。`serve_model` 写、`chat_completion` 读、`unload_model` 删，三者并发时 dict 迭代/修改可能 `RuntimeError: dictionary changed size during iteration` | 偶发 500 |
| REL-10 | 🟠 | `inference.py:18-35` `_cleanup_loaded_models` | **TTL 清理在 MLX unload 失败时仍 `pop` 条目**。`except Exception` 只 log warning，不 return，继续 `_loaded_models.pop`。MLX 端模型还在内存，但本地追踪已丢，永远卸载不掉 | MLX 内存泄漏 |
| REL-11 | 🟡 | `tasks.py:13` `_running_tasks` | **`_running_tasks` dict 的 `add_done_callback` 在事件循环已关闭的关停阶段会抛 `RuntimeError`**，无优雅停机 | 关停噪声 |

### 4.4 可靠性评分明细

| 子项 | 得分 | 说明 |
|------|:----:|------|------|
| 错误处理 | 5/10 | 全局处理器吃异常，错误信息泄露 |
| 重试/退避 | 2/10 | 完全缺失 |
| 事务安全 | 4/10 | 每 CRUD 内部 commit，跨语句无原子性 |
| 竞态条件 | 5/10 | 多处无锁共享状态 |
| 任务生命周期 | 5/10 | 卡死风险、孤儿数据 |
| **小计** | **5.0** | |

---

## 五、可扩展性审计（5.5 / 10，C+）

### 5.1 数据库与连接池

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| SCALE-1 | 🟠 | `db/database.py:14` | **`create_async_engine` 未配置 `pool_size`/`max_overflow`/`pool_recycle`**。仅开了 `pool_pre_ping`。默认 `pool_size=5`，高并发下连接排队 | 高并发瓶颈 |
| SCALE-2 | 🔴 | `db/database.py:14` | **SQLite + aiosqlite 默认 `NullPool`，但代码未显式声明 `check_same_thread=False`**。多 worker（uvicorn `--workers 4`）下 SQLite 写锁竞争会触发 `database is locked` | 多 worker 不可用 |
| SCALE-3 | 🟡 | `app.py:89-95` | **路由注册无前缀分组**。7 个 router 都用 `prefix="/api/v1"`，但内部路径 `/models/...`、`/versions/...` 平铺，无 OpenAPI tag 分组 | API 文档可读性差 |

### 5.2 异步模式

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| SCALE-4 | 🟠 | `storage/local_store.py:34-67` | **`write_chunk`/`assemble_chunks` 标了 `async` 但内部全是同步 `read_bytes`/`write`/`shutil.rmtree`**。5MB chunk × 10000 个的同步 IO 会阻塞事件循环 | 上传期间全服务卡顿 |
| SCALE-5 | 🟡 | `manage/manager.py:50,72` | **`__import__("time").time()`** 这种内联 import + 调用是反模式，且每次调用都重新 import | 代码异味、轻微性能损耗 |

### 5.3 缓存与重复查询

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| SCALE-6 | 🟠 | `system.py:14-50` `health_check` | **每次健康检查都查 DB `list_models(page_size=1)` 拿 total**，无缓存。K8s liveness probe 每秒探一次 = 每秒一次 DB count 查询 | DB 负载 |
| SCALE-7 | 🟠 | `models.py:174` `sync_registry` | **`list_models(page=1, page_size=1000)` 一次性拉全部本地模型**。模型数过千时内存与延迟双杀 | 大规模同步慢 |
| SCALE-8 | 🟡 | `models.py:83-100` `create_model` 后 `session.refresh(m)` | **`selectin` lazy load + refresh 触发 N+1**。`_model_to_dict` 访问 `m.tags` 时才发 SQL，列表接口 N 个模型 = 1 + N 次查询 | 列表慢 |

### 5.4 分页

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| SCALE-9 | 🟡 | `crud.py:14` | `MAX_PAGE_SIZE = 200`，但 `models.py:174` 硬编码 `page_size=1000` 直接绕过上限 | 分页约束被绕过 |
| SCALE-10 | 🟡 | `crud.py:85,176,291,433` | **offset 分页**。深分页（page=1000）时 SQLite 仍要扫描前 20000 行。无 cursor-based 分页 | 深分页慢 |

### 5.5 可扩展性评分明细

| 子项 | 得分 | 说明 |
|------|:----:|------|------|
| 连接池配置 | 4/10 | 未配置 pool 参数 |
| 异步纯度 | 5/10 | storage 同步 IO 阻塞事件循环 |
| 缓存策略 | 3/10 | 完全无缓存，健康检查打 DB |
| 分页 | 7/10 | 有分页但 offset 深分页慢 |
| N+1 查询 | 5/10 | selectin 缓解但 _model_to_dict 触发 tags 懒加载 |
| **小计** | **5.5** | |

---

## 六、内存泄漏风险审计（5.5 / 10，C+）

### 6.1 已识别泄漏点

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| MEM-1 | 🔴 | `versions.py:329` `download_version_from_url` | **`asyncio.create_task` 未追踪、未持有引用**。Python 官方文档明确警告：若不持有 task 引用，任务可能被 GC 中途回收（"Task got destroyed but it is pending"）。且任务异常无人处理，只 log | 下载任务随机消失、异常被吞 |
| MEM-2 | 🟠 | `inference.py:15` `_loaded_models` | **无上限的 dict**。每次 `serve_model` 都 `_loaded_models[model_id] = {...}`，仅靠 TTL 3600s 清理。若攻击者/客户端循环 serve 不同 model_id，dict 无限增长直到 TTL 触发 | 内存增长 |
| MEM-3 | 🟠 | `tasks.py:13,36-37` `_running_tasks` | **`_running_tasks` 用 `add_done_callback` 清理，但回调闭包捕获了 `task_id`**。若任务永不完成（MLX 挂起 600s），dict 条目存活整个超时窗口。高并发 quantize 时多个卡死任务累积 | 任务引用累积 |
| MEM-4 | 🟠 | `storage/local_store.py:37-41` `write_chunk` | **chunk 临时文件无 TTL 清理**。若客户端上传到 chunk 5/10 后断连，`uploads/{upload_id}/*.part` 永远留在磁盘。无定时任务、无启动清理 | 磁盘空间缓慢泄漏 |
| MEM-5 | 🟡 | `inference.py:18-35` `_cleanup_loaded_models` | **TTL 清理只在 `serve_model` 成功后触发**。若长期只有推理调用（不 serve），过期 model 永远不被清 | 清理时机不合理 |
| MEM-6 | 🟡 | `inference.py:117,136,155` | **三个推理端点每次都新建 `AsyncClient`**。高 QPS 下连接池无法复用，TCP 连接频繁建拆 | 连接泄漏 / 性能差 |
| MEM-7 | 🟡 | `auth.py:60-75` 审计日志 | **每次写操作都新建一个 session 写 audit log**。高频写接口（如批量 tag）会额外开 session | session 频繁创建 |

### 6.2 内存泄漏评分明细

| 子项 | 得分 | 说明 |
|------|:----:|------|------|
| 任务追踪 | 3/10 | download_version_from_url 的 task 无引用 |
| 推理模型追踪 | 5/10 | 无上限 dict + 无锁 |
| 临时文件清理 | 4/10 | chunk 临时文件无清理 |
| HTTP 连接复用 | 4/10 | 每次新建 client |
| session 生命周期 | 6/10 | 大部分用 `async with`，但审计日志额外开 session |
| **小计** | **5.5** | |

---

## 七、代码质量与架构审计（6.5 / 10，B-）

### 7.1 代码重复

| ID | 严重度 | 位置 | 问题 |
|----|:------:|------|------|
| CQ-1 | 🟠 | `models.py:138-157`、`versions.py:272-285` | **SSRF 校验逻辑三处重复**（`_is_internal_hostname`、`_validate_url`、`_validate_download_url`），且实现不一致（见 SEC-9） |
| CQ-2 | 🟡 | `inference.py` 三个推理端点 | **chat/completions/embeddings 三个函数体几乎完全相同**（取 info、拼 payload、post、raise），未抽公共方法 |
| CQ-3 | 🟡 | `crud.py` 各 `list_*` 函数 | **`list_models`/`list_versions`/`list_quantize_tasks`/`list_audit_logs` 结构完全一致**（build query、count、offset、limit、execute），可抽象为通用分页 helper |

### 7.2 类型安全

| ID | 严重度 | 位置 | 问题 |
|----|:------:|------|------|
| CQ-4 | 🟠 | `inference.py:117,136,155`、`models.py:230` | **`body: dict` 弱类型**。绕过 Pydantic 校验，调用方可传任意 JSON，`payload = {**body, "model": model_name}` 会把 `body` 里所有字段透传给 MLX（包括恶意的 `max_tokens=10_000_000`） |
| CQ-5 | 🟡 | `models.py:59` `def _model_to_dict(m)` | **`m` 无类型标注**。应是 `m: Model`，当前靠鸭子类型 |
| CQ-6 | 🟡 | `storage/local_store.py:82-88` `is_path_within_store` | **用 `str(resolved).startswith(str(models_resolved))` 做路径包含判断**。`/models/abc` 会误判为在 `/models/ab` 内。应 `resolved.is_relative_to(models_resolved)` 或 `resolved.relative_to(models_resolved)` |
| CQ-7 | 🟡 | `db/models.py:63-64` `_uuid4` | **`uuid.uuid4().hex[:16]` 截断 16 字符**。碰撞概率虽低（16 hex = 64 bit），但截断 UUID 是已知反模式，碰撞后主键冲突 |

### 7.3 关注点分离

| ID | 严重度 | 位置 | 问题 |
|----|:------:|------|------|
| CQ-8 | 🟡 | `server/routers/*.py` | **序列化函数 `_model_to_dict`/`_version_to_dict`/`_node_to_dict` 散落各 router**。应抽到 `server/schemas.py` |
| CQ-9 | 🟡 | `auth.py:99-106` `_extract_resource_id` | **正则硬编码资源类型列表**（`"models", "versions", "quantize"`）。新增 router 时易漏改 |
| CQ-10 | 🟡 | `tasks.py:42-128` `_run_quantize` | **函数 87 行、圈复杂度 8**，混杂了状态更新、版本查找、转换调用、输出创建、错误处理。应拆为 `_mark_running` / `_do_quantize` / `_create_output` / `_mark_completed` |

### 7.4 代码质量评分明细

| 子项 | 得分 | 说明 |
|------|:----:|------|------|
| 类型标注覆盖 | 7/10 | 大部分齐全，少数 `m` 无标注 |
| 重复代码 | 5/10 | SSRF 三处重复、推理三端点重复、CRUD list 重复 |
| 关注点分离 | 6/10 | 序列化未抽离、tasks.py 单函数过大 |
| 命名与可读性 | 8/10 | 命名清晰、注释适度 |
| **小计** | **6.5** | |

---

## 八、系统完整性审计（6.0 / 10，C+）

### 8.1 数据一致性

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| INT-1 | 🔴 | `app.py:18-35,53` | **`_reconcile_orphaned_tasks` 在 `init_deps` 之后、`yield` 之前调用**，但它内部 `from .deps import get_session_factory` + `sf()`。若 `init_deps` 失败或时序变化，启动崩溃。更严重：**它在 `set_auth_enabled` 之前调用**，意味着孤儿任务清理期间认证是关的 | 启动竞态 + 短暂无认证窗口 |
| INT-2 | 🟠 | `models.py:204-214` `batch_delete` | **删 DB model 与删文件分两步、无事务**。`store.delete_model_files(mid)` 成功后 `crud.delete_model` 失败 → 文件没了但 DB 还在；反之 DB 删了文件残留 | 文件-DB 不一致 |
| INT-3 | 🟠 | `versions.py:64-145` 上传流程 | **文件先落盘、DB version 后创建**。若 `create_version` 失败，磁盘上多出孤儿文件。反之无此问题。当前实现是"先文件后DB"，错误方向 | 孤儿文件 |
| INT-4 | 🟡 | `db/models.py:98` `ModelVersion` | **`(model_id, version)` 无联合唯一约束**。`name` 有 `unique=True`，但 version 组合没有，可创建同 model 同 version 的重复行 | 重复数据 |
| INT-5 | 🟡 | `db/models.py:144` `QuantizeTask.output_version_id` | **`output_version_id` 是普通 String(16)，无 ForeignKey**。若输出 version 被删，task 指向悬空 ID | 悬空引用 |

### 8.2 文件处理完整性

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| INT-6 | 🟠 | `storage/local_store.py:43-67` `assemble_chunks` | **组装失败时已写出的部分 `target_path` 不回滚**。若 `chunk_path.read_bytes()` 在第 5 个 chunk 失败，前 4 个已写入 `target_path`，但函数抛异常，调用方 `create_version` 不执行，磁盘残留半成品文件 | 半成品文件残留 |
| INT-7 | 🟠 | `storage/local_store.py:51-59` | **组装时 `hasher.update(data)` 累计 hash，但未在组装后校验 `expected_hash`**。`assemble_chunks` 只返回计算出的 hash，不比对。调用方 `chunk_upload_version` 拿到 hash 后也不校验，直接存 DB | 上传完整性无校验 |
| INT-8 | 🟡 | `storage/local_store.py:90-104` `delete_version_files`/`delete_model_files` | **`shutil.rmtree` 无 `onerror` 回调**。若文件被其他进程占用（MLX 正在用），rmtree 静默失败（`ignore_errors=True`），DB 已删但文件还在 | 静默删失败 |

### 8.3 优雅停机与启动恢复

| ID | 严重度 | 位置 | 问题 | 影响 |
|----|:------:|------|------|------|
| INT-9 | 🔴 | `app.py:47-55` lifespan | **无优雅停机**。`yield` 后无任何清理：`_running_tasks` 里的 quantize task 被 GC 中断、`asyncio.create_task` 的 url-dl task 同样中断、engine 未 `dispose()`、`_loaded_models` 里模型未向 MLX 发 unload | 关停时任务丢失、MLX 端模型残留 |
| INT-10 | 🟠 | `app.py:18-35` `_reconcile_orphaned_tasks` | **孤儿任务清理只在启动时跑一次**，且只清 `RUNNING`/`PENDING`。运行中若进程 OOM 被 kill，重启后能恢复；但若任务在运行中、进程没挂，孤儿清理永远不触发（任务卡死 600s 期间无干预） | 卡死任务无超时清理 |
| INT-11 | 🟡 | `app.py:47-55` lifespan | **`init_db` 失败时 engine 未 dispose**，连接泄漏。lifespan 无 try/finally 包裹 engine 资源 | 启动失败资源泄漏 |

### 8.4 系统完整性评分明细

| 子项 | 得分 | 说明 |
|------|:----:|------|------|
| 数据一致性 | 4/10 | 文件-DB 跨步骤无事务、联合唯一缺失 |
| 文件处理 | 4/10 | 组装失败无回滚、hash 不校验、删失败静默 |
| 优雅停机 | 2/10 | 完全缺失 |
| 启动恢复 | 6/10 | 有孤儿清理但时机/范围有问题 |
| 外键约束 | 7/10 | 大部分 FK 正确，仅 output_version_id 悬空 |
| **小计** | **6.0** | |

---

## 九、跨维度高优先级问题汇总

按"修复性价比"排序（影响 ÷ 修复成本）：

| 排名 | ID | 问题 | 维度 | 修复成本 |
|:----:|----|------|------|:--------:|
| 1 | SEC-8 | 下载重定向 SSRF 绕过 | 安全 | 中 |
| 2 | INT-9 | 无优雅停机 | 完整性 | 中 |
| 3 | SEC-1 | 认证默认关闭 | 安全 | 低 |
| 4 | REL-1 | 零重试/退避 | 可靠性 | 中 |
| 5 | MEM-1 | download task 无引用追踪 | 内存 | 低 |
| 6 | REL-5 | download_version 计数竞态 | 可靠性 | 低 |
| 7 | SEC-2 | 无速率限制 | 安全 | 中 |
| 8 | INT-2/3 | 文件-DB 跨步骤无事务 | 完整性 | 中 |
| 9 | SEC-9/10 | SSRF 三处重复 + 编码绕过 | 安全 | 中 |
| 10 | REL-9/10 | `_loaded_models` 无锁 + 清理时机 | 内存 | 低 |

---

## 十、升级建议路线图

### 第一阶段（安全加固，1-2 天）
- 引入 `server/security.py` 统一安全模块：`validate_url()`、`is_internal_hostname()`、`safe_client()`、`validate_host_header()`
- 删除 `models.py`/`versions.py` 的 SSRF 重复实现，统一调用 security 模块
- 下载流程：重定向后对每个新 host 重新 `validate_url`
- 引入 `slowapi` 或自实现 `RateLimiter`（按 IP token bucket）
- `auth.py` 用 `hmac.compare_digest` 做 key 比较
- API key 改 `hashlib.pbkdf2_hmac('sha256', key, salt, 100_000)` + per-key salt

### 第二阶段（可靠性 + 完整性，2-3 天）
- 引入 `server/reliability.py`：`retry_async` 装饰器（指数退避 + 抖动）、`transactional()` 上下文
- `download_version` 改为"先发文件，成功后异步 +1 计数"或用 `with_for_update` 锁版本行
- `batch_delete`/`upload_version`/`create_version` 用 `transactional()` 包裹跨步骤
- lifespan 增加 `finally` 分支：`await engine.dispose()`、取消所有 `_running_tasks`、向 MLX 发 unload
- `_reconcile_orphaned_tasks` 移到 `init_deps` 之后、`set_auth_enabled` 之前

### 第三阶段（内存 + 可扩展性，2-3 天）
- `_loaded_models` 加 `asyncio.Lock`，`serve_model` 用 double-check
- `download_version_from_url` 的 `create_task` 用 `_running_downloads` dict 持有引用 + `add_done_callback` 清理
- `storage/local_store` 增加 `cleanup_stale_uploads(ttl=3600)`，在 lifespan 启动时跑一次 + 定时任务
- `create_async_engine` 显式配 `pool_size=10, max_overflow=20, pool_recycle=3600`
- `health_check` 结果缓存 5 秒（TTL cache）
- 推理三端点改用 `safe_client` 单例 + 连接池复用

### 第四阶段（代码质量，1-2 天）
- 抽 `server/schemas.py`：所有 `_*_to_dict` 序列化函数集中
- 抽 `server/pagination.py`：通用分页 helper
- `inference.py` 三推理端点抽 `_proxy_inference(model_id, payload, path, timeout)`
- `body: dict` 全部改为 Pydantic `BaseModel` + `Field(ge=, le=)` 校验
- `_uuid4` 改 `uuid.uuid4().hex`（32 字符）或用 `uuid7`

---

## 十一、测试覆盖率与可观测性（附加维度）

> 此维度未纳入综合评分，但影响生产可运维性。

| 项 | 现状 | 评分 |
|----|------|:----:|
| 测试覆盖 | `tests/test_api.py` 97 个 API 集成测试，`test_core.py` 单元测试，`test_coverage.py` 补充 | 7/10 |
| 测试质量 | 用 `httpx.AsyncClient + ASGITransport`，手动 `init_deps`；但无并发测试、无竞态测试 | 6/10 |
| 日志 | `logging.basicConfig` 全局配置，请求日志中间件记录耗时；但无结构化日志（JSON）、无 request-id 贯穿 | 5/10 |
| Metrics | **完全无 Prometheus / OpenMetrics 端点**。无 QPS、延迟分位、错误率、任务积压指标 | 2/10 |
| Tracing | **无 OpenTelemetry / 分布式追踪**。MLX 调用链不可观测 | 2/10 |
| 健康检查 | `/system/health` 区分 `healthy/degraded`，但 `degraded`（MLX 不可用）仍返回 200，K8s liveness 不会重启 | 5/10 |

---

## 十二、结论

**Fusion-Model-Hub 当前处于"功能可用、未达生产级"的状态。** 架构分层清晰、依赖注入规范、字段白名单防越权，这些是良好基础。但安全（无速率限制、SSRF 重定向绕过、认证默认关闭）、可靠性（零重试、竞态条件、无优雅停机）、完整性（文件-DB 跨步骤无事务、组装失败无回滚）三大维度存在多个 🔴 高危问题，阻断直接上生产。

**综合评分：5.9 / 10（C+）**

**升级后预期目标：8.5 / 10（A-）**，即达到"中型团队生产可运维"水平。按本报告第四节路线图，4 个阶段、约 7-10 人日可达成。

---

*报告生成于 2026-07-26，由 AtomCode (GLM-5.2) 基于纯只读代码审计生成，未修改任何代码。*
