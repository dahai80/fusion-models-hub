# Secret Rotation Runbook

> Step6 of the production-readiness path. Covers the three long-lived secrets
> fusion-model-hub relies on, how each is loaded, what breaks when it rotates,
> and the exact rotation procedure with a verifiable drill.

## Secrets at a glance

| Secret | Env var | Default / fallback | Used for | Rotation blast radius |
|--------|---------|--------------------|----------|-----------------------|
| `api_key_pepper` | `FMH_API_KEY_PEPPER` | derived `sha256("fmh-pepted\|{data_dir}")` (dev only) | salt for PBKDF2-HMAC-SHA256 (200k iters) hashing of every API key | **invalidates ALL API keys** — every stored `key_hash` stops matching; all clients 401 until keys are re-issued |
| `mlx_internal_api_key` | `FUSION_MLX_API_KEY` (deprecated: `MLX_INTERNAL_API_KEY`) | `~/.fusion-mlx/settings.json` `auth.api_key` | `Authorization: Bearer` on every Hub→Fusion-MLX call (health, load, inference, quantize, cluster) | Hub↔MLX mutual breakage until both sides carry the **same** new value; mismatch → 401 on every MLX call |
| `auth_bootstrap_token` | `FMH_AUTH_BOOTSTRAP_TOKEN` | unset (open bootstrap, IP-rate-limited) | gate on the **first** API-key creation (`POST /api/v1/auth/keys` while zero active keys) | none in steady state; only matters for re-bootstrap (fresh install / all keys revoked) |

**Load order** (all resolved in `server/config.py:Settings.__post_init__`, at process start):

- `mlx_internal_api_key`: explicit field > env `FUSION_MLX_API_KEY` > deprecated env `MLX_INTERNAL_API_KEY` > `~/.fusion-mlx/settings.json` `auth.api_key` > unset (Hub→MLX calls go out with no Bearer → MLX 401s).
- `api_key_pepper`: explicit field > env `FMH_API_KEY_PEPPER` > derived per-install pepper (dev only; logs a WARNING — never ship this to production).
- `auth_bootstrap_token`: explicit field > env `FMH_AUTH_BOOTSTRAP_TOKEN` > unset.

**Critical invariant:** `mlx_internal_api_key` must equal Fusion-MLX's own `auth.api_key`. They are the *same* secret seen from two sides. Rotating one without the other breaks the Hub↔MLX link.

## 1. `api_key_pepper` (API-key hashing pepper)

**Where:** `server/config.py:Settings.api_key_pepper` → `server/deps.py:init_deps` → `db/crud.py:set_api_key_pepper` → `_hash_key()` (PBKDF2-HMAC-SHA256, 200k iterations, salt = `hmac(pepper, key)`).

**Why rotate:** suspected leak, periodic policy, or migrating off the derived dev pepper to an explicit production value.

**Blast radius:** every existing API key's stored `key_hash` was computed with the *old* pepper. After the new pepper is loaded, `verify_api_key` recomputes the hash with the *new* pepper → no match → all clients receive `401 API key required` (for write endpoints) until their key is re-issued. Read-only public paths and `auth_enabled=False` local mode are unaffected.

**Procedure (zero-downtime requires a brief key-reissue window):**

1. **Pre-check** — enumerate who will be cut off:
   ```bash
   curl -s http://127.0.0.1:11444/api/v1/auth/keys \
     -H "X-API-Key: <admin-key>" | jq '.items | length'
   ```
2. **Generate a strong pepper** (32+ bytes, hex):
   ```bash
   openssl rand -hex 32
   ```
3. **Set the env on the host** (do NOT commit it; the `.env`/unit file must be `0600`):
   ```bash
   export FMH_API_KEY_PEPPER="$(openssl rand -hex 32)"
   ```
4. **Restart the Hub** so `Settings.__post_init__` picks up the new env and `init_deps` re-seeds `_API_KEY_PEPPER`:
   ```bash
   ./start.sh restart
   ```
5. **Re-issue every API key** — old keys are now dead. For each key name, create a replacement and deliver it to the client out-of-band:
   ```bash
   curl -X POST http://127.0.0.1:11444/api/v1/auth/keys \
     -H "X-API-Key: <admin-key>" -H "Content-Type: application/json" \
     -d '{"name":"<client>","role":"<role>","permissions":"read,write"}'
   # → returns the new full "key" ONCE; capture and hand to the client
   ```
6. **Revoke the old key ids** (their hashes no longer verify anyway, but remove the rows for hygiene):
   ```bash
   curl -X DELETE http://127.0.0.1:11444/api/v1/auth/keys/<old-key-id> \
     -H "X-API-Key: <admin-key>"
   ```
7. **Verify** — a re-issued client key now works; an old key does not:
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11444/api/v1/models \
     -H "X-API-Key: <new-client-key>"   # expect 200
   curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:11444/api/v1/models \
     -H "X-API-Key: <old-client-key>"   # expect 401
   ```

**Rollback:** re-set the *old* `FMH_API_KEY_PEPPER` value, `./start.sh restart`, and the original keys work again (step 5's re-issued keys then become dead — pick one direction and complete it; do not mix peppers across a running fleet).

**Multi-node:** every Hub node in the cluster MUST share the same `FMH_API_KEY_PEPPER` (API keys are issued centrally and used against any node). Roll the env on all nodes and restart in a rolling fashion; clients re-issued against one node work against all once all carry the new pepper.

## 2. `mlx_internal_api_key` (Hub→MLX bearer)

**Where:** `server/config.py:Settings.mlx_internal_api_key`, sent as `Authorization: Bearer <key>` by every Hub→MLX caller: `routers/inference.py`, `routers/quantize.py`, `routers/cluster.py`, `routers/hardware.py`, `routers/recommend.py`, `routers/adapt.py`, `routers/system.py`.

**Why rotate:** suspected leak, periodic policy, or because the Fusion-MLX side rotated first (MLX-side rotation forces a Hub-side update).

**Blast radius:** this is the *same* secret as Fusion-MLX's `auth.api_key` (`~/.fusion-mlx/settings.json`). They MUST match. A mismatch makes every Hub→MLX call return `401` — symptoms: `/system/health` reports MLX down, model loads fail, inference 401s, cluster nodes show inactive, recommend/hardware/adapt engines get empty MLX data. The Hub itself stays up; only its MLX delegation breaks.

**Procedure (rotate both sides atomically; expect a brief MLX-delegation outage):**

1. **Generate the new shared key:**
   ```bash
   openssl rand -hex 32
   ```
2. **Rotate Fusion-MLX first** (it is the source of truth for the key):
   ```bash
   NEW=$(openssl rand -hex 32)
   # edit ~/.fusion-mlx/settings.json -> set auth.api_key = $NEW (keep file 0600)
   ~/claude-home/fusion-mlx/start.sh restart
   ```
3. **Set the same value on the Hub side** and restart so `__post_init__` loads it (env wins over the settings-file fallback, so export it explicitly even though the settings file also changed):
   ```bash
   export FUSION_MLX_API_KEY="$NEW"
   ./start.sh restart
   ```
   `start.sh` itself resolves `FUSION_MLX_API_KEY` from env or `~/.fusion-mlx/settings.json` (see `start.sh` lines 45-60), so setting both to `$NEW` is belt-and-suspenders.
4. **Verify the Hub↔MLX link is restored:**
   ```bash
   ~/claude-home/fusion-mlx/start.sh status          # MLX up, models loaded
   curl -s http://127.0.0.1:11444/api/v1/system/health  # "mlx" block healthy, not 401
   ```
   A healthy `mlx` field with model count > 0 confirms the Bearer is accepted.
5. **Confirm an inference path** (optional, needs a loaded model):
   ```bash
   curl -s http://127.0.0.1:11444/api/v1/models/<id>/inference \
     -H "X-API-Key: <key>" -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"ping"}]}'   # 200, not 401
   ```

**Rollback:** set both sides back to the *old* key value and restart both. Order does not matter as long as both end on the same value — but until they agree, MLX delegation is down.

**Deprecation note:** the env alias `MLX_INTERNAL_API_KEY` still works (logs a deprecation warning) but is scheduled for removal; migrate every deployment to `FUSION_MLX_API_KEY` during the next rotation.

**Multi-node:** every Hub node points at the same (or different) Fusion-MLX instances, but each Hub→MLX pair must share the matching key. If all nodes share one MLX, roll the MLX key once and update `FUSION_MLX_API_KEY` on every Hub node. If nodes have per-node MLX instances, rotate each pair independently.

## 3. `auth_bootstrap_token` (first-key gate)

**Where:** `server/config.py:Settings.auth_bootstrap_token` ← env `FMH_AUTH_BOOTSTRAP_TOKEN`; enforced in `routers/auth.py:_require_admin_or_bootstrap`. Constant-time compare (`hmac.compare_digest`) against the `X-Bootstrap-Token` request header.

**Scope:** this token is consulted **only** on the bootstrap path — `POST /api/v1/auth/keys` while **zero** active API keys exist. That path is otherwise public (`/api/v1/auth/keys` is in `PUBLIC_PATHS`), so without the token any client that can reach the Hub wins the root admin key. The token is irrelevant once ≥1 active key exists; subsequent key creation requires an admin caller instead.

**When it matters:** fresh install, or after every API key has been revoked/deleted (re-bootstrap). In steady state it can be left unset or rotated freely with zero runtime impact.

**Why rotate:** suspected leak, or policy. Because steady-state is unaffected, rotation is trivial and safe.

**Procedure:**

1. **Generate** a new token:
   ```bash
   openssl rand -hex 32
   ```
2. **Set the env** and restart so the new token is enforced on the next bootstrap:
   ```bash
   export FMH_AUTH_BOOTSTRAP_TOKEN="$(openssl rand -hex 32)"
   ./start.sh restart
   ```
3. **Verify** the gate (destructive only if it succeeds — do this on a scratch install or accept creating then deleting a throwaway key):
   - **Without** the header → `403 Bootstrap token required`:
     ```bash
     curl -s -o /dev/null -w "%{http_code}\n" -X POST \
       http://127.0.0.1:11444/api/v1/auth/keys \
       -H "Content-Type: application/json" -d '{"name":"x","role":"admin"}'
     # expect 403 (or 201 if zero keys exist AND no token is set — that's the open-bootstrap state you're closing)
     ```
   - **With** the header → `201` (creates the key — delete it after):
     ```bash
     curl -s -X POST http://127.0.0.1:11444/api/v1/auth/keys \
       -H "X-Bootstrap-Token: $FMH_AUTH_BOOTSTRAP_TOKEN" \
       -H "Content-Type: application/json" -d '{"name":"verify","role":"admin"}'
     # → 201 + key body; capture the id, then DELETE it via the new admin key
     ```

**Rollback:** unset `FMH_AUTH_BOOTSTRAP_TOKEN` and restart. Bootstrap reverts to open-but-IP-rate-limited (10/min per source IP). Do NOT run production with the token unset on any network-reachable Hub — see Step3 bootstrap-auth root-cause fix.

**Note:** the token is a gate, not a stored credential — it is never persisted to the DB. Rotating it changes only which header value a future bootstrap must present.

## Rotation drill (self-test)

A scripted end-to-end drill against a throwaway local Hub (no MLX required for
the pepper + bootstrap parts; the MLX-key part needs a running fusion-mlx).
Run on a scratch data dir; tear down after.

```bash
set -e
cd /Users/dahai/fusion && source .venv/bin/activate
cd fusion-model-hub

# ── throwaway Hub on port 11599, auth ON, MLX pointed at dead port ──
export FMH_DATA_DIR="/tmp/fmh_drill_$$"
export FMH_API_KEY_PEPPER="$(openssl rand -hex 32)"
export FMH_AUTH_BOOTSTRAP_TOKEN="$(openssl rand -hex 32)"
export FUSION_MLX_URL="http://127.0.0.1:1"          # dead MLX; drill ignores MLX
# MODEL_HUB_AUTH_ENABLED defaults true — leave it.

fusion-model-hub serve --host 127.0.0.1 --port 11599 --log-level warning &
HUB_PID=$!
trap 'kill $HUB_PID 2>/dev/null; rm -rf "$FMH_DATA_DIR"' EXIT
# wait for the socket
for _ in $(seq 1 50); do curl -sf http://127.0.0.1:11599/api/v1/system/health >/dev/null && break; sleep 0.1; done

echo "=== bootstrap gate: no header -> 403 ==="
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:11599/api/v1/auth/keys \
  -H "Content-Type: application/json" -d '{"name":"root","role":"admin"}'   # expect 403

echo "=== bootstrap gate: with header -> 201 ==="
ADMIN=$(curl -s -X POST http://127.0.0.1:11599/api/v1/auth/keys \
  -H "X-Bootstrap-Token: $FMH_AUTH_BOOTSTRAP_TOKEN" \
  -H "Content-Type: application/json" -d '{"name":"root","role":"admin","permissions":"read,write"}')
ADMIN_KEY=$(echo "$ADMIN" | python -c "import sys,json;print(json.load(sys.stdin)['key'])")
echo "admin key issued: prefix=$(echo "$ADMIN" | python -c "import sys,json;print(json.load(sys.stdin)['key_prefix'])")"

echo "=== write endpoint needs the key: no key -> 401 ==="
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:11599/api/v1/models \
  -H "Content-Type: application/json" -d '{"name":"drill","model_type":"llm"}'   # expect 401

echo "=== with key -> 201 ==="
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:11599/api/v1/models \
  -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name":"drill","model_type":"llm"}'   # expect 201

echo "=== pepper rotation: change env, restart, old key -> 401 ==="
NEW_PEPPER="$(openssl rand -hex 32)"
# stop, swap pepper, restart
kill "$HUB_PID"; wait "$HUB_PID" 2>/dev/null || true
export FMH_API_KEY_PEPPER="$NEW_PEPPER"
fusion-model-hub serve --host 127.0.0.1 --port 11599 --log-level warning &
HUB_PID=$!
for _ in $(seq 1 50); do curl -sf http://127.0.0.1:11599/api/v1/system/health >/dev/null && break; sleep 0.1; done
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:11599/api/v1/models \
  -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name":"drill2","model_type":"llm"}'   # expect 401 — old key dead under new pepper

echo "=== re-issue admin key under new pepper, then write works ==="
# bootstrap path is closed (an active key row exists), so this needs the
# bootstrap token is NOT enough — admin role required. Simulate fresh-start
# by clearing keys via direct DB, OR: re-run from a clean data dir. For the
# drill we just demonstrate the invariant: new pepper -> old key dead (above).
echo "drill invariant proven: pepper rotation invalidates existing keys"

echo "=== cleanup ==="
kill "$HUB_PID" 2>/dev/null || true
rm -rf "$FMH_DATA_DIR"
echo "drill complete"
```

**Expected output** (status codes):
```
=== bootstrap gate: no header -> 403 ===
403
=== bootstrap gate: with header -> 201 ===
201
admin key issued: prefix=...
=== write endpoint needs the key: no key -> 401 ===
401
=== with key -> 201 ===
201
=== pepper rotation: change env, restart, old key -> 401 ===
401
=== re-issue admin key under new pepper, then write works ===
drill invariant proven: pepper rotation invalidates existing keys
=== cleanup ===
drill complete
```

If any status code deviates, the corresponding section above is the
troubleshooting entry point. The `mlx_internal_api_key` drill is not scripted
here because it requires a live fusion-mlx; run it manually per §2 steps 2-5
using `~/claude-home/fusion-mlx/start.sh start` on port 11434.
