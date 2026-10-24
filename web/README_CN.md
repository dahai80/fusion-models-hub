# Fusion-Model-Hub Web 控制台

fusion-model-hub 的 Web UI 包（issue #56，PRD N-4）。四个页面，走 `127.0.0.1:11444` 的 REST API：

- **模型** — 关键词搜索 + 模型表格（30s 轮询）；点击行加载详情面板（版本、标签）；Hugging Face 导入表单从 hf-mirror.com 目录登记仓库（不下载）。
- **下载** — 下载任务表格（5s 轮询）：进度、速度、剩余时间、HF_MIRROR 来源标识、取消（危险色二次确认，规范 §7.3/S5）。速度/ETA 由前端基于连续轮询样本推算——API 只上报累计字节数。
- **量化** — 量化任务提交（源版本 / 格式 / 位宽）+ 任务表格（5s 轮询），状态徽标 + 任务详情面板。
- **存储** — 存储用量卡片（总大小 / 模型数 / 文件数）+ 后端信息与 Fusion-MLX 健康状态（30s 轮询）。

## 鉴权

`MODEL_HUB_AUTH_ENABLED=true`（默认）时写接口需要 `X-API-Key`。**UI 不带任何鉴权逻辑**（规范 S8）——内嵌模式由 shell/gateway 会话在上游注入（manifest `api_key_env` = `FUSION_MODEL_HUB_WEB_API_KEY`）；独立开发模式从 `VITE_HUB_API_KEY` 读取（仅开发便利）。读接口匿名可用。

## 技术栈

React 18 + Vite 5 + TypeScript + Tailwind（fusion preset）+ Bun，复用 `@fusion/web-ui` 组件（StatusBadge、DataTable、MetricTile、KVPanel、ConfirmDialog、EmptyState/ErrorState/Skeleton）与 `@fusion/web-sdk`，通过 `file:vendor/@fusion` 引入——零注册中心依赖。

## 开发

```bash
cd web
bun install
bun run dev        # vite 11475，代理 /api /health /metrics → 11444
bun run build      # typecheck + vite build → dist/
bun run lint       # biome
bun test           # 遥测单元测试
```

`FUSION_DEV_BACKEND` 覆盖后端地址（默认 `http://127.0.0.1:11444`）。`FUSION_MODEL_HUB_WEB_PORT` 覆盖开发端口（已在 `architecture/port-registry.yaml` 登记为 `fusion-model-hub-web`）。

## 网关接线（内嵌模式）

模块启动时向 shell 注册中心自注册 manifest（`web/manifest.json`，id `fusion-models-hub`，分类 `platform`，设备 desktop + mobile-ready）；`/shell/fusion-models-hub/` 代理上游为 :11444 的 Hub API 服务：

```toml
[web_shell.modules.fusion-models-hub]
upstream = "http://127.0.0.1:11444"
```

`?embedded=1` 隐藏独立模式的底部导航——外壳负责整体框架。
