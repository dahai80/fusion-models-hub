import logging
from typing import Any

logger = logging.getLogger(__name__)

# #54: module-level flag (tests toggle without rebuilding Settings); mirrors
# the auth.py set_*_enabled pattern. Initialized at init_deps / lifespan.
_identity_integration_enabled = False


def set_identity_integration_enabled(enabled: bool) -> None:
    global _identity_integration_enabled
    _identity_integration_enabled = bool(enabled)


def is_identity_integration_enabled() -> bool:
    if _identity_integration_enabled:
        return True
    try:
        from .deps import get_settings

        return bool(get_settings().identity_integration_enabled)
    except Exception:
        return False


def _verify_token_sync(identity_url: str, service_token: str, token: str) -> dict[str, Any]:
    # install_tenant_middleware calls verify_jwt(token) synchronously from
    # inside its async __call__, so this runs on the event-loop thread and
    # cannot await. Use a sync httpx.Client. fusion-identity binds 127.0.0.1
    # only (PRD C8), so the call is a loopback round-trip (~sub-ms locally);
    # a 5s timeout bounds any stall. The blocking window is the documented
    # tradeoff of the PRD's sync verify_jwt contract.
    import httpx

    url = f"{identity_url.rstrip('/')}/api/v1/auth/verify"
    logger.debug("identity verify: POST %s", url)
    with httpx.Client(timeout=5.0) as client:
        resp = client.post(
            url,
            json={"token": token},
            headers={"Authorization": f"Bearer {service_token}"},
        )
        if resp.status_code != 200:
            logger.warning(
                "identity verify rejected token: status=%s body=%s",
                resp.status_code,
                resp.text[:200],
            )
            raise ValueError(f"identity verify failed: {resp.status_code}")
        data = resp.json()
    logger.info("identity verify ok: tid=%s role=%s", data.get("tid"), data.get("role"))
    return data


def _make_verify_jwt(identity_url: str, service_token: str):
    def verify_jwt(token: str) -> dict[str, Any]:
        # Resolve via the module global so tests can monkeypatch
        # fusion_model_hub.server.identity._verify_token_sync to a deterministic
        # stub (no real fusion-identity HTTP) without rebuilding the app.
        return _verify_token_sync(identity_url, service_token, token)

    return verify_jwt


def install_identity_middleware(app) -> None:
    from fusion_core.tenant import install_tenant_middleware

    from .auth import PUBLIC_PATHS
    from .deps import get_settings

    settings = get_settings()
    verify_jwt = _make_verify_jwt(settings.identity_url, settings.identity_service_token)
    # Exempt the hub's own public paths (health/docs + the bootstrap
    # /auth/keys endpoint) so liveness probes and first-key creation keep
    # working behind the tenant middleware. fusion-core's default exempt set
    # covers generic /health, /docs, /openapi.json; we add the hub-specific
    # /api/v1/* public paths. Paths are matched exact (rstrip("/")) by the
    # middleware.
    from fusion_core.tenant.middleware import _DEFAULT_EXEMPT

    # #55: in identity-aware mode the JWT `tid` is the authoritative tenant, and
    # API-key provisioning must be tenant-scoped (a caller's key carries their
    # JWT tid). /auth/keys is public in local mode for bootstrap, but in identity
    # mode it must go THROUGH the tenant middleware so the verified tid reaches
    # the route's _caller_tenant — otherwise every key is minted tenant-less and
    # the #55 cross-tenant key check has nothing to compare against. Health/docs
    # stay exempt so liveness probes keep working without a JWT.
    identity_exempt = PUBLIC_PATHS - {"/api/v1/auth/keys"}
    exempt = frozenset(_DEFAULT_EXEMPT | identity_exempt)
    install_tenant_middleware(
        app,
        exempt_paths=exempt,
        verify_jwt=verify_jwt,
        require_jwt=True,
    )
    # Per-app flag read by auth_middleware (request.app.state). Using app state
    # instead of a module global avoids cross-test-file flag leakage: each app
    # instance carries its own integration setting, so an identity-enabled test
    # app cannot affect the auth path of a separate identity-disabled app.
    app.state.identity_integration_enabled = True
    logger.info(
        "fusion-identity tenant middleware installed: url=%s issuer=%s exempt=%d",
        settings.identity_url,
        settings.identity_jwt_issuer,
        len(exempt),
    )


def current_tenant_context():
    # #54: thin accessor over fusion_core.tenant.current() so auth.py does not
    # import fusion_core directly (keeps the integration dep isolated here).
    # Returns the TenantContext set by TenantMiddleware, or None when the
    # request was exempt / integration disabled.
    try:
        from fusion_core.tenant.context import current

        return current()
    except Exception:
        return None
