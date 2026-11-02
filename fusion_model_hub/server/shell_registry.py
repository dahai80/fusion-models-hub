import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

GATEWAY_URL_ENV = "FUSION_SHELL_GATEWAY_URL"
MANIFEST_PATH_ENV = "FMH_SHELL_MANIFEST"
UPSTREAM_ENV = "FMH_SHELL_UPSTREAM"
KEY_FILE_NAME = "web_module_key"
REGISTER_INTERVAL_SECONDS = 15.0
HTTP_TIMEOUT_SECONDS = 5.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_manifest_path() -> Path:
    return _repo_root() / "web" / "manifest.json"


def default_web_dist() -> Path:
    return _repo_root() / "web" / "dist"


def module_key_file(data_dir: str) -> Path:
    return Path(data_dir) / KEY_FILE_NAME


def load_module_key(data_dir: str) -> str:
    try:
        return module_key_file(data_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_module_key(data_dir: str, key: str) -> bool:
    key = (key or "").strip()
    if not key:
        return False
    try:
        path = module_key_file(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key + "\n", encoding="utf-8")
        path.chmod(0o600)
        logger.info("gateway-issued module key persisted: %s", path)
        return True
    except OSError as e:
        logger.warning("module key file unwritable: %s", e)
        return False


def shell_accept_keys(data_dir: str) -> list[str]:
    # The gateway /shell proxy strips the browser's Authorization and injects
    # the fm- key it holds for this module as Bearer (same contract as the
    # trainer bridge). Direct loopback access keeps using X-API-Key. The
    # FUSION_CLIENT_KEY chain is the console contract credential (§5.1).
    keys: list[str] = []
    try:
        from fusion_core.config import client_key

        ck = client_key().strip()
        if ck:
            keys.append(ck)
    except ImportError:
        pass
    issued = load_module_key(data_dir)
    if issued and issued not in keys:
        keys.append(issued)
    return keys


def _extract_issued_key(payload: Any) -> str:
    # Gateway K1 issuance returns the plaintext exactly once in `raw_key`
    # (writeShellKeyResponse, fusion-gateway internal/server/shell.go).
    if not isinstance(payload, dict):
        return ""
    for field in ("raw_key", "api_key", "issued_key", "key", "module_key"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = payload.get("data")
    if isinstance(nested, dict):
        for field in ("raw_key", "api_key", "issued_key", "key", "module_key"):
            value = nested.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def resolve_upstream(manifest: dict[str, Any]) -> str:
    upstream = (
        os.environ.get(UPSTREAM_ENV, "").strip()
        or str(manifest.get("upstream") or "").strip()
        or "http://127.0.0.1:11444"
    )
    parsed = urlparse(upstream)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"shell upstream must be an absolute http(s) URL: {upstream}")
    return upstream.rstrip("/")


def build_registration_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    # No api_key in the body — that is the K1 trigger: the gateway issues the
    # fm- service key instead of accepting a module-supplied credential.
    payload = dict(manifest)
    payload["upstream"] = resolve_upstream(manifest)
    payload.pop("api_key", None)
    return payload


def load_manifest(path: str | None = None) -> dict[str, Any]:
    manifest_path = Path(path or os.environ.get(MANIFEST_PATH_ENV, "") or default_manifest_path())
    if not manifest_path.exists():
        raise FileNotFoundError(f"shell manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("id"):
        raise ValueError(f"shell manifest missing id: {manifest_path}")
    logger.info("shell manifest loaded from %s (id=%s)", manifest_path, manifest["id"])
    return manifest


def _gateway_admin_key() -> str:
    # Single-point resolution via fusion_core.config.admin_key (env
    # FUSION_ADMIN_KEY; prod-missing raises, dev-missing warns + ""). A raise
    # here must never take the hub down — degrade to disabled registration.
    try:
        from fusion_core.config import admin_key

        return admin_key().strip()
    except RuntimeError as e:
        logger.warning("shell registration disabled: %s", e)
        return ""
    except ImportError:
        return ""


def shell_registration_enabled() -> bool:
    gateway = os.environ.get(GATEWAY_URL_ENV, "").strip()
    return bool(gateway and _gateway_admin_key())


def _safe_json(resp: Any) -> Any:
    try:
        return resp.json()
    except Exception:
        return None


async def _registry_loop_body(
    client: Any,
    manifest: dict[str, Any],
    data_dir: str,
    interval: float,
    max_ticks: int | None = None,
) -> None:
    module_id = str(manifest["id"])
    gateway = os.environ.get(GATEWAY_URL_ENV, "").strip().rstrip("/")
    admin_key = _gateway_admin_key()
    admin_headers = {
        "Authorization": f"Bearer {admin_key}",
        "Content-Type": "application/json",
    }
    try:
        payload = build_registration_payload(manifest)
    except Exception as e:
        logger.error("shell registration disabled (manifest unusable): %s", e)
        return
    logger.info("shell manifest upstream=%s api_key=unset (K1 gateway-issued)", payload["upstream"])
    registered = False
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        try:
            if not registered:
                resp = await client.post(
                    f"{gateway}/api/v1/shell/manifests",
                    json=payload,
                    headers=admin_headers,
                )
                if resp.status_code == 200:
                    registered = True
                    logger.info("shell manifest registered: %s", module_id)
                    issued = _extract_issued_key(_safe_json(resp))
                    if issued and issued != load_module_key(data_dir):
                        save_module_key(data_dir, issued)
                    elif issued:
                        logger.info("gateway-issued module key unchanged, not rewriting file")
                    else:
                        logger.info(
                            "registration retained existing service key — plaintext "
                            "shown only at issuance (rotate via gateway admin if lost)"
                        )
                else:
                    logger.warning("shell manifest register failed: %d %s", resp.status_code, resp.text[:200])
            else:
                module_key = load_module_key(data_dir)
                beat_key = module_key or admin_key
                headers = {
                    "Authorization": f"Bearer {beat_key}",
                    "Content-Type": "application/json",
                }
                resp = await client.post(
                    f"{gateway}/api/v1/shell/manifests/{module_id}/heartbeat",
                    headers=headers,
                )
                if resp.status_code == 404:
                    # Gateway restarted and dropped dynamic manifests.
                    logger.info("shell heartbeat 404, re-registering %s", module_id)
                    registered = False
                elif resp.status_code in (401, 403) and module_key:
                    # Issued key rotated/revoked — re-register via admin key
                    # to pick up the fresh issued key.
                    logger.warning(
                        "shell heartbeat %d (issued key invalid), re-registering %s", resp.status_code, module_id
                    )
                    registered = False
                elif resp.status_code != 200:
                    logger.warning("shell heartbeat failed: %d %s", resp.status_code, resp.text[:200])
        except asyncio.CancelledError:
            logger.info("shell registry loop cancelled (module=%s)", module_id)
            raise
        except Exception as e:
            logger.warning("shell registry loop error (will retry): %s", e)
        if max_ticks is None or ticks < max_ticks:
            await asyncio.sleep(interval)


async def shell_registry_loop(manifest_path: str | None = None, data_dir: str = "") -> None:
    # Register once, heartbeat forever; re-register after gateway restarts.
    # Runs as a daemon background task from the FastAPI lifespan. All
    # failures are logged and retried on the next tick — never fatal.
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        logger.error("shell registration disabled (manifest unusable): %s", e)
        return
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        await _registry_loop_body(client, manifest, data_dir, REGISTER_INTERVAL_SECONDS)
