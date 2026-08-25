import logging
import time

from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from ..db.crud import verify_api_key
from ..db.models import ApiKey, UserRole
from .deps import get_session_factory

logger = logging.getLogger(__name__)

WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
DELETE_METHODS = {"DELETE"}
# Precise path match (no startswith) — /auth/keys/{id} sub-paths must auth.
# POST /auth/keys is public ONLY via bootstrap guard (route checks zero active keys).
PUBLIC_PATHS = {
    "/api/v1/system/health",
    "/docs",
    "/openapi.json",
    "/api/v1/auth/keys",
}

# In-process timestamp cache: throttle last_used_at DB writes to once per N sec per
# key, so verify stays a pure read (F-08). Falls back to write when entry stale.
_LAST_USED_THROTTLE_SECONDS = 30
_last_used_cache: dict[str, float] = {}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _is_public_path(path: str) -> bool:
    # Exact match or explicit sub-path of a public entry only when the public
    # entry is a directory-style prefix (trailing-slash safe). /auth/keys itself
    # is public for bootstrap POST but its /{id} sub-paths are NOT — so match
    # /auth/keys exactly, never its children.
    if path in PUBLIC_PATHS:
        return True
    return any(public.endswith("/") and path.startswith(public) for public in PUBLIC_PATHS)


async def get_current_api_key(request: Request, api_key: str = Depends(api_key_header)) -> ApiKey | None:
    if not api_key:
        return None
    session_factory = get_session_factory()
    async with session_factory() as session:
        return await verify_api_key(session, api_key)


def _check_role_permission(role: UserRole, method: str) -> JSONResponse | None:
    if role == UserRole.ADMIN:
        return None
    if role == UserRole.DEVELOPER:
        if method in DELETE_METHODS:
            return JSONResponse(status_code=403, content={"detail": "Developer role cannot perform DELETE operations"})
        return None
    if role == UserRole.VIEWER:
        if method in WRITE_METHODS:
            return JSONResponse(status_code=403, content={"detail": "Viewer role is read-only"})
        return None
    return JSONResponse(status_code=403, content={"detail": "Unknown role"})


# E-S15: path segments that can legally follow /models when they are a
# collection-level operation rather than a model_id. The prior skip-set was
# ("import","sync","batch","compare") — incomplete: /models/import/hf,
# /models/batch/delete, /models/batch/tag, /models/recommend, /models/search,
# /models/market/search all extracted the keyword (or skipped it and found no
# later "models") and returned "" → _check_model_access returned None → a
# restricted key bypassed its allowed_models ACL on exactly the operations
# (import, batch delete/tag) that create or remove models outside its scope.
# Treat every segment in this set as "collection op, no single model_id".
_MODEL_COLLECTION_KEYWORDS = {
    "import", "sync", "batch", "compare", "recommend", "search",
    "market", "tag", "delete", "publish-all",
}


def _extract_model_id_from_path(path: str) -> str:
    parts = path.strip("/").split("/")
    for i, p in enumerate(parts):
        if p == "models" and i + 1 < len(parts):
            candidate = parts[i + 1]
            if candidate in _MODEL_COLLECTION_KEYWORDS:
                # Collection op — no single model_id to scope against. Signal
                # this distinctly from "not found" so callers can deny a
                # restricted key rather than silently skip the ACL.
                return ""
            return candidate
    return ""


def _is_model_collection_op(path: str) -> bool:
    parts = path.strip("/").split("/")
    for i, p in enumerate(parts):
        if p == "models" and i + 1 < len(parts):
            return parts[i + 1] in _MODEL_COLLECTION_KEYWORDS
    return False


def _check_model_access(ak: ApiKey, request: Request) -> JSONResponse | None:
    if not ak.allowed_models:
        return None
    allowed_set = {x.strip() for x in ak.allowed_models.split(",") if x.strip()}
    if not allowed_set:
        return None
    path = request.url.path
    # E-S15: a restricted key (non-empty allowed_models) must not reach
    # collection-level model operations — they have no single model_id to
    # scope against, so the prior code silently skipped the ACL and let a
    # model-scoped key POST /models/import/hf or /models/batch/delete outside
    # its scope. Deny explicitly.
    if _is_model_collection_op(path):
        return JSONResponse(
            status_code=403,
            content={"detail": "Collection model operations are not permitted for model-scoped API keys"},
        )
    model_id = _extract_model_id_from_path(path)
    if not model_id:
        return None
    if model_id not in allowed_set:
        return JSONResponse(status_code=403, content={"detail": f"Model '{model_id}' not allowed for this API key"})
    return None


def _check_module_access(ak: ApiKey, request: Request) -> JSONResponse | None:
    if not ak.allowed_modules:
        return None
    allowed_set = {x.strip() for x in ak.allowed_modules.split(",") if x.strip()}
    if not allowed_set:
        return None
    module = request.headers.get("X-Fusion-Module", "")
    if not module:
        return None
    module = module.lower().strip()
    if module not in allowed_set:
        return JSONResponse(status_code=403, content={"detail": f"Module '{module}' not allowed for this API key"})
    return None


async def auth_middleware(request: Request, call_next):
    if "/audit" in request.url.path and request.method == "DELETE":
        return JSONResponse(status_code=403, content={"detail": "Audit logs cannot be deleted"})

    # F-01/#58: a "public" path means no key is REQUIRED, not that a presented
    # key is ignored. Before, the early return here skipped authentication
    # entirely on public paths, so request.state.user_role was never set even
    # when a valid admin key was sent — POST /auth/keys then 403'd any admin
    # creating a 2nd key (route reads _caller_role which came back ""). Now:
    # public only relaxes the no-key rejection; a presented key is still
    # verified and its role/tenant stamped onto request.state so the route's
    # admin-or-bootstrap guard works for both bootstrap (anonymous) and
    # post-bootstrap (admin key) callers.
    is_public = _is_public_path(request.url.path)

    api_key_str = request.headers.get("X-API-Key", "")
    if not api_key_str:
        if is_public or not _is_auth_enabled():
            return await call_next(request)
        return JSONResponse(status_code=401, content={"detail": "API key required"})

    session_factory = get_session_factory()
    async with session_factory() as session:
        ak = await verify_api_key(session, api_key_str)
        if not ak:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        # Role ACL applies to all methods (F-04.5): viewer blocked from writes;
        # developer blocked from delete; admin full. GET passes for all roles.
        if request.method in WRITE_METHODS:
            role_denied = _check_role_permission(ak.role, request.method)
            if role_denied:
                return role_denied

        # Model/module ACL enforced on every method, not only writes (F-04.5):
        # a restricted key must not GET an out-of-ACL model.
        model_denied = _check_model_access(ak, request)
        if model_denied:
            return model_denied

        module_denied = _check_module_access(ak, request)
        if module_denied:
            return module_denied

        # Throttled last_used_at refresh (F-08): verify_api_key is a pure read;
        # we touch the DB at most once per _LAST_USED_THROTTLE_SECONDS per key.
        now = time.time()
        last = _last_used_cache.get(ak.id, 0.0)
        if now - last >= _LAST_USED_THROTTLE_SECONDS:
            _last_used_cache[ak.id] = now
            try:
                from ..db.crud import touch_api_key_last_used
                await touch_api_key_last_used(session, ak.id)
            except Exception:
                logger.debug("last_used_at refresh failed for key %s", ak.id, exc_info=True)

        from .rate_limit import check_rate_limit
        if not check_rate_limit(ak.key_prefix, ak.qps_limit):
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        request.state.api_key_id = ak.id
        request.state.api_key_name = ak.name
        request.state.tenant_id = ak.tenant_id
        request.state.user_role = ak.role.value

    response = await call_next(request)

    # E-S12: audit-log write failure previously swallowed silently (logger.exception
    # only). A DB hiccup could drop the audit record of a write that already
    # succeeded — undetectable. Now emit a structured WARNING naming the action +
    # resource so a lost audit row is visible in logs (the op still succeeds to
    # avoid a DB outage blocking all writes).
    try:
        from ..db.crud import create_audit_log
        resource_type = _extract_resource_type(request.url.path)
        resource_id = _extract_resource_id(request.url.path)
        action = f"{request.method.lower()}_{resource_type}"
        if "/inference/" in request.url.path:
            pass
        else:
            async with session_factory() as session:
                await create_audit_log(
                    session,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    api_key_id=getattr(request.state, "api_key_id", ""),
                    tenant_id=getattr(request.state, "tenant_id", ""),
                    detail=f"{request.method} {request.url.path}",
                )
    except Exception:
        logger.warning(
            "AUDIT LOG LOST: action=%s resource=%s/%s key=%s tenant=%s status=%d",
            f"{request.method.lower()}_{_extract_resource_type(request.url.path)}",
            _extract_resource_type(request.url.path),
            _extract_resource_id(request.url.path),
            getattr(request.state, "api_key_id", ""),
            getattr(request.state, "tenant_id", ""),
            response.status_code,
            exc_info=True,
        )

    return response


_auth_enabled = False


def set_auth_enabled(enabled: bool):
    global _auth_enabled
    _auth_enabled = enabled


def _is_auth_enabled() -> bool:
    return _auth_enabled


def _extract_resource_type(path: str) -> str:
    parts = path.strip("/").split("/")
    if len(parts) >= 4:
        return parts[3]
    return "unknown"


def _extract_resource_id(path: str) -> str:
    parts = path.strip("/").split("/")
    for i, p in enumerate(parts):
        if p in ("models", "versions", "quantize") and i + 1 < len(parts):
            candidate = parts[i + 1]
            if candidate not in (
                "import", "download-url", "chunk-upload",
                "running", "deprecate", "retire", "rollback", "benchmark", "status",
            ):
                return candidate
    return ""
