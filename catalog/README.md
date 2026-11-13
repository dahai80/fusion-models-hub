---
doc_id: fusion-models-hub/catalog
product: fusion-models-hub
class: core
entry: service
doc_type: reference
title: { zh: "静态模型目录格式", en: "Static model catalog format" }
audience: engineer
status: draft
version: 1.2.21
updated: 2026-11-13
owner: "@dahai"
source_of_truth:
  - catalog/models/
  - fusion_model_hub/catalog/schema.py
  - ci/validate_catalog.py
---

# Static Model Catalog

Issue [#102](https://github.com/dahai80/fusion-models-hub/issues/102) — one declarative JSON file per model under `models/`, versioned in-repo. Static files beat runtime discovery for cross-node consistency: the catalog is lintable, diffable, and consumable by every fusion-* project without querying a live hub.

## File format (`schemaVersion: 1`)

```json
{
  "schemaVersion": 1,
  "name": "mlx-community--Qwen3-8B-4bit",
  "hubId": "",
  "revision": "",
  "contextWindowTokens": 32768,
  "maxOutputTokens": 8192,
  "pricing": {
    "currency": "USD",
    "inputPerMillionTokens": 0.0,
    "outputPerMillionTokens": 0.0
  },
  "modality": { "input": ["text"], "output": ["text"] },
  "capabilities": { "functionCall": true, "reasoning": true, "vision": false, "embedding": false },
  "quantizations": ["4bit"]
}
```

| Field | Rules |
|---|---|
| `schemaVersion` | must be `1` |
| `name` | non-empty; file name is `<name>.json` with `/` escaped as `__` |
| `contextWindowTokens` / `maxOutputTokens` | `> 0`; `maxOutputTokens <= contextWindowTokens` |
| `pricing` | per-million-token prices; zero is valid (local serving) — the field exists for hybrid setups |
| `modality.input` / `modality.output` | non-empty lists; values from `text`, `image`, `audio`, `video` |
| `capabilities` | boolean flags: `functionCall`, `reasoning`, `vision`, `embedding` |
| `quantizations` | free-form labels, e.g. `4bit`, `8bit`, `bf16` |

Unknown fields are rejected (fail loudly on drift). Validator: `python ci/validate_catalog.py [dir]` (exit 1 on any invalid file). The hub also validates on read — `GET /api/v1/catalog` serves only entries that pass the schema.

## Maintenance

Add or edit a file per model, then run the validator. CI runs the validator on every push so schema drift fails the build. Re-seeding from a live fusion-mlx is possible via `/v1/models` (names only — the metadata here is curated by hand).
