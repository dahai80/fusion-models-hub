from __future__ import annotations

import asyncio
import json
import logging

import typer

logger = logging.getLogger(__name__)

keys_app = typer.Typer(no_args_is_help=True)


async def _setup_db():
    # #107: key management runs offline against the hub's own DB. Settings()
    # resolves DATABASE_URL/FMH_* env exactly like the server does, including
    # the api_key_pepper derivation — a key minted here verifies server-side.
    from ..db.database import get_engine, init_db
    from ..server.config import Settings
    from ..server.deps import init_deps

    settings = Settings()
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    from ..db import crud

    crud.set_api_key_pepper(settings.api_key_pepper)
    return engine


@keys_app.command("create")
def keys_create(
    name: str = typer.Option(..., help="Key name, e.g. fusion-trainer"),
    role: str = typer.Option("developer", help="admin | developer | viewer"),
    tenant_id: str = typer.Option("", help="Bind the key to a tenant id (empty = root)"),
    qps_limit: int = typer.Option(0, help="QPS limit (0 = unlimited)"),
):
    # #107: documented provisioning path for service callers. The full key is
    # printed ONCE (only its hash is stored) — pipe it straight into the
    # caller's env, e.g. FUSION_MODEL_HUB_API_KEY for fusion-trainer.
    ak, full_key = asyncio.run(_create(name, role, tenant_id, qps_limit))
    typer.echo(
        json.dumps(
            {
                "id": ak.id,
                "name": ak.name,
                "role": ak.role.value,
                "key_prefix": ak.key_prefix,
                "key": full_key,
                "note": "Store this key now — it is not recoverable later",
            },
            indent=2,
        )
    )


async def _create(name: str, role: str, tenant_id: str, qps_limit: int):
    from ..db import crud
    from ..db.database import get_session_factory

    engine = await _setup_db()
    if role not in ("admin", "developer", "viewer"):
        raise typer.BadParameter(f"role must be admin|developer|viewer, got {role!r}")

    async with get_session_factory(engine)() as session:
        # FK guard: api_keys.tenant_id references tenants — the server-side
        # create fails the same way, so the CLI surfaces it before minting.
        if tenant_id:
            from ..db.models import Tenant

            tenant = await session.get(Tenant, tenant_id)
            if tenant is None:
                raise typer.BadParameter(f"tenant not found: {tenant_id!r}")
        ak, full_key = await crud.create_api_key(
            session,
            name=name,
            tenant_id=tenant_id,
            role=role,
            qps_limit=qps_limit,
        )
    logger.info("Provisioned API key via CLI: id=%s role=%s", ak.id, role)
    return ak, full_key


@keys_app.command("list")
def keys_list(
    tenant_id: str = typer.Option("", help="Filter by tenant id"),
):
    rows = asyncio.run(_list(tenant_id))
    if not rows:
        typer.echo("No API keys")
        return
    for r in rows:
        typer.echo(f"{r['id']}  {r['key_prefix']}...  role={r['role']}  active={r['active']}  {r['name']}")


async def _list(tenant_id: str):
    from ..db import crud
    from ..db.database import get_session_factory

    engine = await _setup_db()
    async with get_session_factory(engine)() as session:
        keys = await crud.list_api_keys(session, tenant_id=tenant_id)
        return [
            {
                "id": k.id,
                "key_prefix": k.key_prefix,
                "role": k.role.value,
                "active": k.is_active,
                "name": k.name,
            }
            for k in keys
        ]


@keys_app.command("deactivate")
def keys_deactivate(
    key_id: str = typer.Argument(..., help="Key id (see fmh keys list)"),
):
    ok = asyncio.run(_deactivate(key_id))
    if ok:
        typer.echo(f"Deactivated: {key_id}")
    else:
        typer.echo(f"Key not found: {key_id}")
        raise typer.Exit(code=1)


async def _deactivate(key_id: str):
    from ..db import crud
    from ..db.database import get_session_factory

    engine = await _setup_db()
    async with get_session_factory(engine)() as session:
        ak = await crud.get_api_key(session, key_id)
        if not ak:
            return False
        await crud.deactivate_api_key(session, key_id)
        return True
