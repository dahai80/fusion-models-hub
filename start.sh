#!/usr/bin/env bash
# start.sh — Fusion-Model-Hub API 服务启动脚本
# 供 Fusion-Studio UpstreamServiceManager 调用：./start.sh start|stop|restart|status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.fusion-model-hub.pid"
LOG_DIR="$SCRIPT_DIR/logs"
STDOUT_LOG="$LOG_DIR/stdout.log"
STDERR_LOG="$LOG_DIR/stderr.log"
HOST="${FUSION_MODEL_HUB_HOST:-127.0.0.1}"
PORT="${FUSION_MODEL_HUB_PORT:-11444}"
MLX_URL="${FUSION_MLX_URL:-http://127.0.0.1:11434}"
# P1-13: TLS support. When FMH_TLS_CERTFILE and FMH_TLS_KEYFILE are set, pass
# them to `serve` so uvicorn terminates TLS directly. Lets an operator run the
# Hub on a network without a reverse proxy fronting it. Without a gateway, raw
# API keys over plain HTTP are sniffable on the wire; TLS closes that.
TLS_CERT="${FMH_TLS_CERTFILE:-}"
TLS_KEY="${FMH_TLS_KEYFILE:-}"

mkdir -p "$LOG_DIR"

is_running() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null
}

resolve_venv() {
    # 优先用本仓库根目录共享 .venv
    local root_venv="$SCRIPT_DIR/../.venv"
    if [ -d "$root_venv" ]; then
        # shellcheck disable=SC1091
        source "$root_venv/bin/activate"
    elif [ -d "$SCRIPT_DIR/.venv" ]; then
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
}

resolve_mlx_api_key() {
    # Hub→MLX requests carry Authorization: Bearer <key> (config.py
    # mlx_internal_api_key). MLX rejects keyless callers with 401 when its
    # own settings.json auth.api_key is set — so without this the hub's
    # /v1/models/status, /v1/chat/completions, /load and cluster sync calls
    # all silently fail (empty mlx_metrics, model never loads). Resolve the
    # same way fusion-mlx/start.sh does: explicit env wins, else read
    # ~/.fusion-mlx/settings.json auth.api_key, else leave unset.
    # #74: canonical name FUSION_CLIENT_KEY only — the legacy alias chain
    # lives in fusion_core.config and nowhere else. config.py picks this up
    # via client_key() when the MLX-native sources yield nothing.
    if [ -n "${FUSION_CLIENT_KEY:-}" ]; then
        return 0
    fi
    local mlx_settings="$HOME/.fusion-mlx/settings.json"
    if [ -f "$mlx_settings" ]; then
        local key
        key="$(python3 -c "import json; print(json.load(open('$mlx_settings')).get('auth',{}).get('api_key',''))" 2>/dev/null || true)"
        if [ -n "$key" ]; then
            export FUSION_CLIENT_KEY="$key"
            echo "FUSION_CLIENT_KEY resolved from $mlx_settings"
            return 0
        fi
    fi
    echo "WARNING: FUSION_CLIENT_KEY unset and ~/.fusion-mlx/settings.json has no auth.api_key — Hub→MLX calls will 401" >&2
}

check_port_registry() {
    # #74 Phase 3.3: verify this service's port-registry.yaml entry before
    # serving, so an unregistered port fails at start instead of at CI.
    # Missing registry file = warn + continue (standalone deploy);
    # FUSION_SKIP_REGISTRY_CHECK=1 bypasses the refusal.
    # Registry key is "fusion-model-hub" (no "s") — same alias mapping as
    # scripts/check-ports.py.
    local registry="${FUSION_REGISTRY:-$HOME/fusion/architecture/port-registry.yaml}"
    if [ ! -f "$registry" ]; then
        echo "warn: port-registry.yaml not found at $registry — registry self-check skipped" >&2
        return 0
    fi
    if grep -A3 "^  fusion-model-hub:" "$registry" 2>/dev/null | grep -q "port: $PORT$"; then
        return 0
    fi
    if [ "${FUSION_SKIP_REGISTRY_CHECK:-}" != "1" ]; then
        echo "port $PORT for fusion-models-hub not registered in $registry — refusing to start" >&2
        return 1
    fi
    echo "warn: registry check bypassed (FUSION_SKIP_REGISTRY_CHECK=1)" >&2
}

do_start() {
    if is_running; then
        echo "fusion-model-hub already running (pid $(cat "$PID_FILE"))"
        return 0
    fi
    resolve_venv
    check_port_registry
    resolve_mlx_api_key
    local serve_args=(serve --host "$HOST" --port "$PORT" --mlx-url "$MLX_URL")
    if [ -n "$TLS_CERT" ] && [ -n "$TLS_KEY" ]; then
        serve_args+=(--tls-certfile "$TLS_CERT" --tls-keyfile "$TLS_KEY")
        echo "TLS enabled: cert=$TLS_CERT"
    fi
    nohup fusion-model-hub "${serve_args[@]}" \
        >> "$STDOUT_LOG" 2>> "$STDERR_LOG" &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 2
    if is_running; then
        echo "fusion-model-hub started (pid $pid, port $PORT)"
        return 0
    else
        echo "fusion-model-hub failed to start, see $STDERR_LOG" >&2
        rm -f "$PID_FILE"
        return 1
    fi
}

do_stop() {
    if ! is_running; then
        echo "fusion-model-hub not running"
        rm -f "$PID_FILE"
        return 0
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    # R-P2/#13: graceful drain. SIGTERM lets uvicorn stop accepting new
    # connections and finish in-flight requests (long chat completions can run
    # many seconds). Before, the loop polled only 5x1s then unconditionally
    # SIGKILL'd — a slow inference was cut mid-response and the client got a
    # truncated stream. Now poll up to DRAIN_TIMEOUT (default 30s) and only
    # escalate to SIGKILL if the process is still alive after the drain window.
    local drain_timeout="${FMH_DRAIN_TIMEOUT:-30}"
    kill -TERM "$pid" 2>/dev/null || true
    echo "fusion-model-hub drain started (pid $pid, timeout ${drain_timeout}s)"
    local waited=0
    while [ "$waited" -lt "$drain_timeout" ]; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "fusion-model-hub did not exit after ${drain_timeout}s drain, sending SIGKILL" >&2
        kill -9 "$pid" 2>/dev/null || true
    else
        echo "fusion-model-hub drained and exited cleanly after ${waited}s"
    fi
    rm -f "$PID_FILE"
    echo "fusion-model-hub stopped"
}

do_status() {
    if is_running; then
        echo "fusion-model-hub running (pid $(cat "$PID_FILE"), port $PORT)"
        return 0
    else
        echo "fusion-model-hub not running"
        return 1
    fi
}

case "${1:-start}" in
    start)  do_start ;;
    stop)   do_stop ;;
    restart) do_stop || true; do_start ;;
    status) do_status ;;
    *) echo "Usage: $0 {start|stop|restart|status}" >&2; exit 1 ;;
esac
