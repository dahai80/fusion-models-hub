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
    "/api/v1/system/storage",
    "/api/v1/auth/keys",
    "/api/v1/cluster/nodes",
    "/api/v1/models/sync",
    "/api/v1/models/batch",
    "/api/v1/models/compare",
    "/metrics",
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


async def auth_middleware(request: Request, call_next):
    if request.method not in WRITE_METHODS:
        return await call_next(request)

    for public in PUBLIC_PATHS:
        if request.url.path.startswith(public):
            return await call_next(request)

    api_key_str = request.headers.get("X-API-Key", "")
    if not api_key_str:
        auth_enabled = _is_auth_enabled()
        if not auth_enabled:
            return await call_next(request)
        return JSONResponse(status_code=401, content={"detail": "API key required"})

    session_factory = get_session_factory()
    async with session_factory() as session:
        ak = await verify_api_key(session, api_key_str)
        if not ak:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        role_denied = _check_role_permission(ak.role, request.method)
        if role_denied:
            return role_denied

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
