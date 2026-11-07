---
doc_id: fusion-models-hub/product-page
product: fusion-models-hub
class: core
entry: service
doc_type: product-page
title: { zh: "fusion-models-hub 产品页", en: "fusion-models-hub product page" }
audience: engineer
status: draft
version: 1.2.15
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - pyproject.toml
  - fusion_model_hub/server/app.py
---

# fusion-models-hub

```yaml
slug: models-hub
name: fusion-models-hub
tagline:
  zh: 本地模型仓库中枢：注册、量化、版本管理、LoRA 合并与基准评估，一个服务管全部
  en: Local model-registry hub: registration, quantization, versioning, LoRA merge, and benchmarking in one service
cta:
  zh: 三条命令把本地 MLX 模型纳入统一仓库与版本管理
  en: Bring local MLX models under one registry with three commands
repo: https://github.com/dahai80/fusion-models-hub
```

## 定位

```yaml
product:
  type: core-service
  serves: 需要模型目录/版本/量化生命周期的 fusion 组件（web-shell 工作台、mlux 客户端、运维脚本）
  builds_on:
    - fusion-mlx（推理引擎 :11434）与 fusion-gateway（:11432）：连接探测与 LLM 转发
    - fusion-core（HTTP/配置/日志底座）
  hosts: 模型元数据仓库、量化/LoRA 任务调度、版本晋升与回滚、基准记录
  integration_surface: HTTP API（:11444，X-API-Key 鉴权）+ Python SDK（fusion_model_hub.sdk.client）
  deployment: 本机或内网单实例；SQLite（默认）或 PostgreSQL；存储 local 或 MinIO
```

## capabilities

```yaml
capabilities:
  - icon: 🗂
    name: { zh: 模型注册与目录, en: Model registry and catalog }
    tag:
      zh: 本地/HF 模型统一入库，应用方不各自维护模型清单
      en: Local and HF models in one catalog; apps stop keeping their own model lists
    points:
      - zh: 手动注册、HF 导入（import_from_hf）、注册表同步（sync_registry）
        en: Manual registration, HF import, and registry sync
      - zh: 约束：目录是元数据与文件登记，不做模型格式转换
        en: Constraint: the catalog tracks metadata and files, not format conversion
    viz: []
  - icon: 🧮
    name: { zh: 量化流水线, en: Quantization pipeline }
    tag:
      zh: 4/8-bit 量化任务后台跑，应用方不写转换脚本
      en: Background quantization jobs; no hand-written conversion scripts
    points:
      - zh: start_quantize / layered quantize / evaluate_quantize 全链路，任务状态可查
        en: start_quantize through layered quantize and evaluation, with queryable task status
      - zh: 约束：量化耗时随模型规模线性增长，大模型任务需轮询 get_quantize_status
        en: Constraint: runtime scales with model size; poll get_quantize_status for large jobs
    viz: []
  - icon: 🔖
    name: { zh: 版本管理与晋升, en: Version management and promotion }
    tag:
      zh: 版本生命周期（promote/rollback/deprecate/retire）有单一权威实现
      en: One authoritative implementation of the version lifecycle (promote/rollback/deprecate/retire)
    points:
      - zh: 版本对比（compare_models）、基准记录（benchmark）、回滚
        en: Model comparison, benchmark records, and rollback
      - zh: 约束：回滚改指向不迁移数据，旧版本文件由存储层负责保留
        en: Constraint: rollback repoints, it does not migrate; storage keeps old version files
    viz: []
  - icon: 🔀
    name: { zh: LoRA 合并, en: LoRA merge }
    tag:
      zh: LoRA 适配器合并成独立模型，应用方不用碰底层权重
      en: Merge LoRA adapters into standalone models without touching weights by hand
    points:
      - zh: start_lora_merge / get_lora_merge_status 异步任务
        en: Async jobs via start_lora_merge and get_lora_merge_status
      - zh: 约束：合并产物为新的完整模型，占用额外存储
        en: Constraint: the merged output is a full new model and consumes extra storage
    viz: []
  - icon: 🔌
    name: { zh: OpenAI 兼容转发, en: OpenAI-compatible proxy }
    tag:
      zh: chat/completions 经 Hub 转发到 MLX，客户端只对一个地址
      en: chat/completions proxied through the hub to MLX; clients face one address
    points:
      - zh: 依赖 FMH_MLX_URL / FMH_MLX_TOOL_URL 指向网关或引擎
        en: Backed by FMH_MLX_URL / FMH_MLX_TOOL_URL pointing at gateway or engine
      - zh: 约束：MLX 未连接时转发端点不可用，仓库功能不受影响（见 health.mlxConnected）
        en: Constraint: proxy endpoints need MLX reachable; catalog works without it (see health.mlxConnected)
    viz: []
  - icon: 🖥
    name: { zh: Web 控制台, en: Web console }
    tag:
      zh: 内置控制台管理模型与任务，不用手写 curl
      en: Built-in console for models and tasks; no hand-written curl
    points:
      - zh: FMH_WEB_DIST_DIR 指向构建产物即可挂载
        en: Mount by pointing FMH_WEB_DIST_DIR at a built bundle
      - zh: 约束：控制台与 API 共用同一鉴权，无匿名模式
        en: Constraint: the console shares API auth; there is no anonymous mode
    viz: []
```

## workflow

```yaml
workflow:
  - name: { zh: 启动服务, en: Start service }
    action: { zh: ./start.sh start 或 fusion-model-hub serve, en: ./start.sh start or fusion-model-hub serve }
    input: { zh: 环境配置（.env 或环境变量）, en: Environment config (.env or env vars) }
    output: { zh: :11444 健康端点 200, en: Health endpoint on :11444 returns 200 }
  - name: { zh: 建模型目录, en: Build catalog }
    action: { zh: 注册/导入模型，跑量化与 LoRA 任务, en: Register or import models; run quantization and LoRA jobs }
    input: { zh: 模型文件或 HF 仓库 id, en: Model files or HF repo ids }
    output: { zh: 带版本的模型清单, en: Versioned model catalog }
  - name: { zh: 发布与回滚, en: Promote and roll back }
    action: { zh: 基准评估后晋升版本，问题版本回滚, en: Benchmark, promote versions, roll back bad ones }
    input: { zh: 基准结果（benchmark 端点）, en: Benchmark results (benchmark endpoint) }
    output: { zh: 稳定版本指针, en: Stable version pointer }
  - name: { zh: 供下游消费, en: Serve consumers }
    action: { zh: SDK 或 HTTP 供 shell/客户端查询与转发, en: SDK or HTTP for shell/client queries and proxying }
    input: { zh: X-API-Key, en: X-API-Key }
    output: { zh: 统一模型访问入口, en: One unified model access point }
```

## useCases

```yaml
useCases:
  - name: { zh: web-shell 工作台模型管理, en: Model management in the web-shell workbench }
    headline: { zh: 工作台的模型列表/切换来自 Hub 目录，而非本地扫描, en: The workbench reads its model list from the hub catalog instead of scanning locally }
    body:
      zh: shell 通过 FUSION_SHELL_GATEWAY_URL 自注册清单，经 SDK 读目录。差别：多端看到同一份模型状态，切换动作有审计。
      en: The shell self-registers its manifest and reads the catalog via the SDK. Difference: every client sees the same model state, and switches are auditable.
  - name: { zh: 量化实验管理, en: Quantization experiment management }
    headline: { zh: 同一模型的多个量化版本并排对比, en: Multiple quantized variants of one model compared side by side }
    body:
      zh: 量化任务产出带版本记录，compare_models + benchmark 留存数据。差别：选型靠记录不靠记忆。
      en: Quantization jobs produce versioned records; compare_models and benchmark retain the data. Difference: decisions from records, not memory.
```

## techDepth

```yaml
techDepth:
  - category: { zh: 可靠性纪律, en: Reliability disciplines }
    items:
      - name: { zh: 健康自检, en: Health self-check }
        desc: { zh: /api/v1/system/health 报告版本、MLX 连接、存储水位, en: /api/v1/system/health reports version, MLX connectivity, and storage }
        steps:
          - zh: start.sh 启动后自检该端点, en: start.sh checks this endpoint after startup
      - name: { zh: 密钥轮换, en: Secret rotation }
        desc: { zh: API key 轮换有 runbook, en: API key rotation has a runbook }
        steps:
          - zh: 见 docs/secret-rotation-runbook.md, en: See docs/secret-rotation-runbook.md
```

## limits（已知边界）

- zh: 单实例部署，无内建多副本/分布式；鉴权依赖自管 key，无 OAuth/SSO
- en: Single-instance deployment, no built-in replication; auth is self-managed keys, no OAuth/SSO

## 提交检查清单

- [x] 所有字段 zh/en 成对
- [x] 每条能力回答"应用方少写了什么"
- [x] integration_surface 与仓库实际一致（HTTP + SDK，来自 README）
- [x] 端口 11444 在 port-registry 注册
- [x] 无营销形容词
