import logging

from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from ..db.crud import verify_api_key
from ..db.models import ApiKey, UserRole
from .deps import get_session_factory

logger = logging.getLogger(__name__)

WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
DELETE_METHODS = {"DELETE"}
PUBLIC_PATHS = {
    "/api/v1/system/health",
    "/docs",
    "/openapi.json",
    "/api/v1/auth/keys",
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


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


def _extract_model_id_from_path(path: str) -> str:
    parts = path.strip("/").split("/")
    for i, p in enumerate(parts):
        if p == "models" and i + 1 < len(parts):
            candidate = parts[i + 1]
            if candidate not in (
                "import", "sync", "batch", "compare",
            ):
                return candidate
    return ""


def _check_model_access(ak: ApiKey, request: Request) -> JSONResponse | None:
    if not ak.allowed_models:
        return None
    allowed_set = {x.strip() for x in ak.allowed_models.split(",") if x.strip()}
    if not allowed_set:
        return None
    model_id = _extract_model_id_from_path(request.url.path)
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

    for public in PUBLIC_PATHS:
        if request.url.path.startswith(public):
            return await call_next(request)

    api_key_str = request.headers.get("X-API-Key", "")
    if not api_key_str:
        if not _is_auth_enabled():
            return await call_next(request)
        return JSONResponse(status_code=401, content={"detail": "API key required"})

    session_factory = get_session_factory()
    async with session_factory() as session:
        ak = await verify_api_key(session, api_key_str)
        if not ak:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        if request.method in WRITE_METHODS:
            role_denied = _check_role_permission(ak.role, request.method)
            if role_denied:
                return role_denied

            model_denied = _check_model_access(ak, request)
            if model_denied:
                return model_denied

            module_denied = _check_module_access(ak, request)
            if module_denied:
                return module_denied

        from .rate_limit import check_rate_limit
        if not check_rate_limit(ak.key_prefix, ak.qps_limit):
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        request.state.api_key_id = ak.id
        request.state.api_key_name = ak.name
        request.state.tenant_id = ak.tenant_id
        request.state.user_role = ak.role.value

    response = await call_next(request)

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
        logger.exception("Failed to write audit log")

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
