# Fusion-Model-Hub Web Console

Web UI package for fusion-model-hub (issue #56, PRD N-4). Four pages over
the REST API on `127.0.0.1:11444`:

- **Models** — keyword search + model table (30s poll); row click loads the
  detail panel (versions, tags); Hugging Face import form registers a repo
  from the hf-mirror.com catalog without downloading.
- **Downloads** — download task table (5s poll) with progress, speed, ETA,
  HF_MIRROR source badge and cancel behind a danger confirm (spec §7.3/S5).
  Speed/ETA are derived client-side from successive poll samples — the API
  reports cumulative bytes only.
- **Quantize** — quantize task submit (source version / format / bits) +
  task table (5s poll) with status badges and per-task detail panel.
- **Storage** — storage usage tiles (total size / models / files) + backend
  info and Fusion-MLX health (30s poll).

## Auth

The hub requires `X-API-Key` on write endpoints when
`MODEL_HUB_AUTH_ENABLED=true` (default). **The UI ships no auth logic**
(spec S8) — embedded mode the shell/gateway session injects the key
upstream via the `/shell/fusion-models-hub/` proxy (manifest `api_key_env` =
`FUSION_MODEL_HUB_WEB_API_KEY`); standalone dev wires it from
`VITE_HUB_API_KEY` (dev convenience only). Read endpoints work anonymously.

## Stack

React 18 + Vite 5 + TypeScript + Tailwind (fusion preset) + Bun, sharing
`@fusion/web-ui` components (StatusBadge, DataTable, MetricTile, KVPanel,
ConfirmDialog, EmptyState/ErrorState/Skeleton) and `@fusion/web-sdk` via
`file:vendor/@fusion` — zero registry dependencies.

## Dev

```bash
cd web
bun install
bun run dev        # vite on 11475, proxies /api /health /metrics → 11444
bun run build      # typecheck + vite build → dist/
bun run lint       # biome
bun test           # telemetry unit tests
```

`FUSION_DEV_BACKEND` overrides the backend target (default
`http://127.0.0.1:11444`). `FUSION_MODEL_HUB_WEB_PORT` overrides the dev
port (registered as `fusion-model-hub-web` in
`architecture/port-registry.yaml`).

## Gateway wiring (embedded mode)

The module self-registers its manifest (`web/manifest.json`, id
`fusion-models-hub`, category `platform`, devices desktop + mobile-ready)
with the shell registry at startup; the `/shell/fusion-models-hub/` proxy
upstream is the hub API server on `:11444`:

```toml
[web_shell.modules.fusion-models-hub]
upstream = "http://127.0.0.1:11444"
```

`?embedded=1` hides the standalone bottom nav — the shell owns the chrome.
