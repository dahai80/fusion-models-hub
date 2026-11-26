from __future__ import annotations

import pytest

from fusion_model_hub.cli import keys as keys_cli


@pytest.fixture()
def keys_env(tmp_path, monkeypatch):
    # Each helper builds its own Settings()/engine, so the DB must be a real
    # file (in-memory sqlite dies with each engine) under an isolated data dir.
    db_file = tmp_path / "keys.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setenv("FMH_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


@pytest.mark.asyncio()
async def test_create_list_roundtrip(keys_env):
    ak, full_key = await keys_cli._create("fusion-trainer", "developer", "", 0)
    assert full_key.startswith(f"fmh-{ak.key_prefix.split('-', 1)[1]}-")
    rows = await keys_cli._list("")
    assert [r["id"] for r in rows] == [ak.id]
    assert rows[0]["role"] == "developer"
    assert rows[0]["active"] is True


@pytest.mark.asyncio()
async def test_created_key_verifies_with_same_pepper(keys_env):
    # Core #107 guarantee: a key minted by the CLI (Settings-derived pepper)
    # must verify through the same code path the server uses.
    from fusion_model_hub.db import crud
    from fusion_model_hub.db.database import get_engine, get_session_factory

    _, full_key = await keys_cli._create("verify-me", "admin", "", 0)
    engine = get_engine(f"sqlite+aiosqlite:///{keys_env}/keys.db")
    async with get_session_factory(engine)() as session:
        verified = await crud.verify_api_key(session, full_key)
        assert verified is not None
        assert verified.role.value == "admin"
        assert await crud.verify_api_key(session, "fmh-0000-bogus") is None


@pytest.mark.asyncio()
async def test_deactivate_removes_from_active_flow(keys_env):
    ak, _ = await keys_cli._create("to-die", "viewer", "", 0)
    assert await keys_cli._deactivate(ak.id) is True
    assert await keys_cli._deactivate("nope") is False
    rows = await keys_cli._list("")
    assert rows[0]["active"] is False


@pytest.mark.asyncio()
async def test_tenant_filter(keys_env):
    from fusion_model_hub.db.database import get_engine, get_session_factory
    from fusion_model_hub.db.models import Tenant

    engine = get_engine(f"sqlite+aiosqlite:///{keys_env}/keys.db")
    await keys_cli._setup_db()
    async with get_session_factory(engine)() as session:
        session.add(Tenant(id="tenant-a", name="Tenant A"))
        session.add(Tenant(id="tenant-b", name="Tenant B"))
        await session.commit()

    a, _ = await keys_cli._create("t-a", "developer", "tenant-a", 0)
    await keys_cli._create("t-b", "developer", "tenant-b", 0)
    rows = await keys_cli._list("tenant-a")
    assert [r["id"] for r in rows] == [a.id]


@pytest.mark.asyncio()
async def test_unknown_tenant_rejected(keys_env):
    from typer import BadParameter

    with pytest.raises(BadParameter):
        await keys_cli._create("t-x", "developer", "ghost-tenant", 0)


@pytest.mark.asyncio()
async def test_bad_role_rejected(keys_env):
    from typer import BadParameter

    with pytest.raises(BadParameter):
        await keys_cli._create("bad", "superuser", "", 0)
