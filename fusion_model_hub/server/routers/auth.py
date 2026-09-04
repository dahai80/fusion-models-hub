import logging
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ..auth import _is_auth_enabled
from ..deps import SessionDep, SettingsDep
from ..rate_limit import check_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])

# E-E6: bootstrap (first-key) path is public, so throttle it per source IP to
# stop a flood of racing root-key creations. 10/min is generous for a legit
# single bootstrap but blocks a scripted race. Node-local like all rate_limit.
_BOOTSTRAP_QPS = 10


class ApiKeyCreate(BaseModel):
    name: str
    tenant_id: str = ""
    permissions: str = "read,write"
    role: str = Field("developer", pattern="^(admin|developer|viewer)$")
    qps_limit: int = 0
    allowed_models: str = ""
    allowed_modules: str = ""


def _caller_tenant(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or ""


# #55: fusion-identity JWT roles (tenant_admin / admin / developer / member /
# viewer) are the role source in identity-aware mode, while local mode uses the
# UserRole enum value (admin / developer / viewer). Normalize the identity role
# space onto the local admin/developer/viewer ladder so the create-key route's
# admin-or-bootstrap guard and the tenanted-admin pinning work under both. A
# role that does not map is treated as viewer (least privilege, fail-closed).
_IDENTITY_ROLE_MAP = {
    "tenant_admin": "admin",
    "admin": "admin",
    "developer": "developer",
    "member": "viewer",
    "viewer": "viewer",
}


def _caller_role(request: Request) -> str:
    role = getattr(request.state, "user_role", "") or ""
    if role in _IDENTITY_ROLE_MAP:
        return _IDENTITY_ROLE_MAP[role]
    return role


def _client_ip(request: Request) -> str:
    # E-E6: behind a proxy X-Forwarded-For carries the real client; fall back to
    # the direct connection. Used only as a rate-limit bucket key for bootstrap.
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _require_admin_or_bootstrap(session, request: Request, settings) -> None:
    # F-01: POST /auth/keys is public ONLY for the very first key (bootstrap).
    # Once any active key exists, creating more requires an admin caller.
    # auth_enabled=False (local mode) bypasses — matches existing local semantics.
    if not _is_auth_enabled():
        return
    active_count = await crud.count_active_api_keys(session)
    if active_count == 0:
        # E-E6: bootstrap is public, so harden it: (1) IP rate-limit the
        # endpoint so a race of N concurrent root-key POSTs from one source
        # cannot stampede, and (2) if the operator set FMH_AUTH_BOOTSTRAP_TOKEN,
        # require a matching X-Bootstrap-Token header — otherwise any client
        # that can reach the Hub wins root. Token is a constant-time compare.
        ip = _client_ip(request)
        if not check_rate_limit(f"bootstrap:{ip}", _BOOTSTRAP_QPS):
            logger.warning("Bootstrap rate-limited for ip=%s", ip)
            raise HTTPException(status_code=429, detail="Bootstrap rate limit exceeded, retry shortly")
        expected = getattr(settings, "auth_bootstrap_token", "") or ""
        if expected:
            supplied = request.headers.get("X-Bootstrap-Token", "")
            import hmac

            if not supplied or not hmac.compare_digest(supplied, expected):
                logger.warning("Bootstrap rejected: missing/invalid X-Bootstrap-Token for ip=%s", ip)
                raise HTTPException(status_code=403, detail="Bootstrap token required to create the first key")
        logger.info("Bootstrap: creating first admin key (no active keys present) ip=%s", ip)
        return
    role = _caller_role(request)
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admin can create API keys once bootstrap key exists",
        )


@router.post("/auth/keys", status_code=201)
async def create_key(body: ApiKeyCreate, session: SessionDep, request: Request, settings: SettingsDep):
    await _require_admin_or_bootstrap(session, request, settings)
    # P1-12: cross-tenant key forge. Before, body.tenant_id always won, so any
    # admin could mint a key for an arbitrary tenant. Now a tenanted admin is
    # forced to its own tenant and may NOT set a different one; a root/super
    # admin (no tenant) and bootstrap keep provisioning any tenant. Local mode
    # (auth off) preserves the legacy body-or-empty behavior.
    caller_tenant = _caller_tenant(request)
    caller_role = _caller_role(request)
    if not _is_auth_enabled():
        tenant_id = body.tenant_id or ""
    elif caller_role == "admin" and not caller_tenant:
        # root/super-admin provisioning: may target any tenant
        tenant_id = body.tenant_id or ""
    elif caller_role == "admin" and caller_tenant:
        # tenanted admin: pinned to own tenant, cross-tenant body value rejected
        if body.tenant_id and body.tenant_id != caller_tenant:
            logger.warning(
                "Cross-tenant key forge blocked: caller_tenant=%s body_tenant=%s",
                caller_tenant,
                body.tenant_id,
            )
            raise HTTPException(
                status_code=403,
                detail="Cannot create an API key for a different tenant",
            )
        tenant_id = caller_tenant
    else:
        # bootstrap (first key): root admin, no tenant
        tenant_id = ""
    ak, full_key = await crud.create_api_key(
        session,
        name=body.name,
        tenant_id=tenant_id,
        permissions=body.permissions,
        role=body.role,
        qps_limit=body.qps_limit,
        allowed_models=body.allowed_models,
        allowed_modules=body.allowed_modules,
    )
    return {
        "id": ak.id,
        "name": ak.name,
        "tenant_id": ak.tenant_id,
        "key": full_key,
        "key_prefix": ak.key_prefix,
        "permissions": ak.permissions,
        "role": ak.role.value,
        "qps_limit": ak.qps_limit,
        "allowed_models": ak.allowed_models,
        "allowed_modules": ak.allowed_modules,
        "is_active": ak.is_active,
        "created_at": ak.created_at.isoformat() if ak.created_at else None,
    }


@router.get("/auth/keys")
async def list_keys(session: SessionDep, request: Request):
    # F-04: scope to caller tenant. Empty tenant (local/unset) returns all,
    # preserving auth-disabled local-mode semantics.
    tenant_id = _caller_tenant(request)
    keys = await crud.list_api_keys(session, tenant_id=tenant_id)
    return {
        "items": [
            {
                "id": k.id,
                "name": k.name,
                "tenant_id": k.tenant_id,
                "key_prefix": k.key_prefix,
                "permissions": k.permissions,
                "role": k.role.value,
                "qps_limit": k.qps_limit,
                "allowed_models": k.allowed_models,
                "allowed_modules": k.allowed_modules,
                "is_active": k.is_active,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            }
            for k in keys
        ]
    }


@router.delete("/auth/keys/{key_id}")
async def delete_key(key_id: str, session: SessionDep):
    # F-01: sub-path never public; auth middleware enforces caller identity.
    deleted = await crud.delete_api_key(session, key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted", "id": key_id}


@router.post("/auth/keys/{key_id}/deactivate")
async def deactivate_key(key_id: str, session: SessionDep):
    ak = await crud.deactivate_api_key(session, key_id)
    if not ak:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"id": ak.id, "is_active": ak.is_active}


@router.get("/auth/keys/{key_id}/usage")
async def key_usage(key_id: str, session: SessionDep, request: Request):
    ak = await crud.get_api_key(session, key_id)
    if not ak:
        raise HTTPException(status_code=404, detail="API key not found")
    tenant_id = _caller_tenant(request)
    if tenant_id and ak.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="API key not found")

    # E-E7: aggregate ONLY this key's inference volume. The prior version summed
    # every entry in _model_stats (all keys, all tenants) into one total — so
    # any key with access to /auth/keys/{id}/usage saw every other key's
    # request counts and tokens, a cross-tenant/same-tenant business-intel
    # leak. _update_model_stats now records a per_key breakdown keyed by
    # api_key_id; filter to ak.id and fall back to the "" (anonymous, auth-off
    # local mode) bucket only when this key itself is anonymous-equivalent.
    from .inference import _model_stats

    total_requests = 0
    by_model: dict[str, dict] = {}
    last_at = None
    for model_id, stats in _model_stats.items():
        per_key = stats.get("per_key", {})
        key_bucket = per_key.get(ak.id)
        if not key_bucket:
            continue
        count = key_bucket.get("request_count", 0)
        if count <= 0:
            continue
        total_requests += count
        by_model[model_id] = {
            "request_count": count,
            "total_tokens": key_bucket.get("total_tokens", 0),
            "avg_latency_ms": round(key_bucket.get("total_latency", 0.0) / count, 2) if count else 0.0,
        }
        la = stats.get("last_request_at")
        if la and (last_at is None or la > last_at):
            last_at = la

    qps_current = 0.0
    if last_at:
        last_dt = datetime.fromtimestamp(last_at, tz=UTC)
        elapsed = (datetime.now(UTC) - last_dt).total_seconds()
        if elapsed < 60:
            qps_current = round(total_requests / max(elapsed, 1.0), 2)

    return {
        "key_id": key_id,
        "total_requests": total_requests,
        "qps_current": qps_current,
        "qps_limit": ak.qps_limit,
        "last_used": ak.last_used_at.isoformat() if ak.last_used_at else None,
        "by_model": by_model,
    }
