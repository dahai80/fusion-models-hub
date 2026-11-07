---
doc_id: fusion-models-hub/get-started
product: fusion-models-hub
class: core
entry: service
doc_type: get-started
title: { zh: "fusion-models-hub 快速上手", en: "fusion-models-hub quick start" }
audience: engineer
status: draft
version: 1.2.15
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - start.sh
  - .env.example
---

# fusion-models-hub 快速上手（Get Started）

> 目标：5–10 分钟让模型仓库服务在本机跑起来并完成第一次 API 调用。

## 1. 前置条件

- macOS（Apple Silicon 生态内使用），Python 3.12（根 `.venv`）
- 可选：PostgreSQL（不配则用 SQLite 开发默认）；fusion-mlx / fusion-gateway 在跑时 Hub 自动探测连接
- 端口：`11444`（已在 `~/fusion/architecture/port-registry.yaml` 注册）

## 2. 安装

```bash
# fusion-core 不在 PyPI，先装 git tag
pip install "fusion-core @ git+https://github.com/dahai80/fusion-core.git@v0.10.0"
pip install -e ".[test]"
```

确认点：

```bash
fusion-model-hub --help | head -5
# 期望：列出 serve 等子命令
```

## 3. 启动

```bash
./start.sh start                  # 推荐：注册端口 + 启动 + 自检
# 或直接：
fusion-model-hub serve --host 127.0.0.1 --port 11444
```

确认点（`/api/v1/system/health` 为公开端点，本机实测输出）：

```bash
curl -s localhost:11444/api/v1/system/health
# {"status":"healthy","version":"1.2.15","uptime":"14s","mlxConnected":true,
#  "model_count":0,"mlx":{"status":"available","url":"http://127.0.0.1:11432",...},
#  "storage":{"used":0.0,...}}
```

- 日志：`logs/`（start.sh 模式）
- 停止：`./start.sh stop`

## 4. 第一次调用

鉴权默认开启（`FMH_AUTH_ENABLED=true`）。无 key 请求返回：

```json
{"detail":"API key required"}
```

API key 存在 Hub 数据库内，经管理流程创建（引导令牌与轮换见
[config-reference](config-reference.md) 与 [secret-rotation-runbook](secret-rotation-runbook.md)）。
持有效 key 后（头名 `X-API-Key`）：

```bash
curl -s localhost:11444/api/v1/models -H "X-API-Key: <your-key>"
# 期望：模型列表 JSON（分页结构）
```

SDK 等价：

```python
from fusion_model_hub.sdk.client import FusionModelHubClient
client = FusionModelHubClient(base_url="http://127.0.0.1:11444", api_key="<your-key>")
print(client.list_models())
```

## 5. 下一步

- 全部端点：[api-reference](api-reference.md)
- 全部配置：[config-reference](config-reference.md)
- 架构：[architecture](architecture.md)
- Web 控制台：README 的 Web Console 一节（`web/dist` 构建）

## 6. 常见问题（最多 3 条）

```yaml
troubleshooting:
  - q: { zh: "API key required / Invalid API key", en: "API key required / Invalid API key" }
    a: { zh: "鉴权默认开启且 key 存在 Hub 自己的数据库，环境变量里的网关 key 不通用。经管理流程创建后用 X-API-Key 头。", en: "Auth is on by default and keys live in the Hub's own database; gateway keys from env do not work. Create a key via the admin flow and send the X-API-Key header." }
  - q: { zh: "health 里 mlxConnected:false", en: "mlxConnected:false in health" }
    a: { zh: "fusion-mlx/gateway 未运行。Hub 仍可用（仓库功能正常），LLM 转发类端点不可用。", en: "fusion-mlx/gateway is not running. The hub still works (catalog features); LLM-proxy endpoints do not." }
  - q: { zh: "启动报数据库连接错误", en: "Database connection error on startup" }
    a: { zh: "DATABASE_URL 未建库或 URL 未被规范化。先 createdb fusion_model_hub；裸 postgresql:// 会被规范化为 postgresql+asyncpg://。", en: "Database not created or URL not normalized. Run createdb fusion_model_hub first; bare postgresql:// URLs are normalized to postgresql+asyncpg://." }
```

## 提交检查清单

- [x] serve/health/模型列表命令本机实测，输出真实
- [x] 每步有确认点
- [x] 端口 11444 在 port-registry 注册
- [x] zh/en 成对（命令除外）
- [x] 全篇 ≤ 120 行
