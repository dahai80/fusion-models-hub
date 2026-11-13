import pytest
from httpx import ASGITransport, AsyncClient

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.server.app import create_app
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import init_deps

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


@pytest.fixture
def settings():
    return Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_data",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        models_dir="/tmp/fmh_test_data/models",
        eval_runner_enabled=False,
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app, settings):
    from fusion_model_hub.server.auth import set_auth_enabled

    set_auth_enabled(False)
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _body(name, sha, **kw):
    payload = {"name": name, "sha256": sha}
    payload.update(kw)
    return payload


class TestDatasetRegistration:
    async def test_register_and_get(self, client):
        resp = await client.post(
            "/api/v1/datasets",
            json=_body("eval-export-1108", SHA_A, num_samples=1200, size_bytes=4096),
        )
        assert resp.status_code == 200
        d = resp.json()
        assert d["name"] == "eval-export-1108"
        assert d["sha256"] == SHA_A
        assert d["num_samples"] == 1200
        got = await client.get(f"/api/v1/datasets/{d['id']}")
        assert got.status_code == 200
        assert got.json()["id"] == d["id"]

    async def test_register_duplicate_409_with_existing_id(self, client):
        first = (await client.post("/api/v1/datasets", json=_body("dup", SHA_A))).json()
        resp = await client.post("/api/v1/datasets", json=_body("dup", SHA_A))
        assert resp.status_code == 409
        assert first["id"] in resp.json()["detail"]

    async def test_register_rejects_bad_sha256(self, client):
        resp = await client.post("/api/v1/datasets", json=_body("bad-sha", "nothex"))
        assert resp.status_code == 422

    async def test_get_missing_404(self, client):
        resp = await client.get("/api/v1/datasets/doesnotexist")
        assert resp.status_code == 404


class TestDatasetList:
    async def test_list_and_keyword(self, client):
        await client.post("/api/v1/datasets", json=_body("rollouts-batch-1", SHA_A))
        await client.post("/api/v1/datasets", json=_body("cowork-export-2", SHA_B))
        resp = await client.get("/api/v1/datasets")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2
        resp = await client.get("/api/v1/datasets", params={"keyword": "rollouts"})
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["name"] == "rollouts-batch-1"

    async def test_lineage_stored_round_trip(self, client):
        lineage = {
            "source": "from-eval",
            "source_path": "/tmp/export.jsonl",
            "source_sha256": SHA_C,
            "transforms": [{"op": "dedup", "at": "2026-11-13T08:00:00Z"}],
            "output": "/tmp/dataset.jsonl",
        }
        d = (
            await client.post(
                "/api/v1/datasets",
                json=_body("lineage-ds", SHA_A, lineage=lineage),
            )
        ).json()
        got = (await client.get(f"/api/v1/datasets/{d['id']}")).json()
        assert got["lineage"]["source"] == "from-eval"
        assert got["lineage"]["transforms"][0]["op"] == "dedup"


class TestDatasetLineageChain:
    async def test_chain_resolves_ancestors(self, client):
        # grandparent (export) -> parent (sft mix) -> child (rollout mix):
        # each child's lineage.source_sha256 points at the parent file sha.
        await client.post(
            "/api/v1/datasets",
            json=_body("export-root", SHA_A, lineage={"source": "from-eval", "source_sha256": SHA_C}),
        )
        await client.post(
            "/api/v1/datasets",
            json=_body("sft-mix", SHA_B, lineage={"source": "from-eval", "source_sha256": SHA_A}),
        )
        child = (
            await client.post(
                "/api/v1/datasets",
                json=_body("rollout-mix", SHA_C, lineage={"source": "from-rollouts", "source_sha256": SHA_B}),
            )
        ).json()
        resp = await client.get(f"/api/v1/datasets/{child['id']}/lineage")
        assert resp.status_code == 200
        chain = resp.json()["chain"]
        assert [c["name"] for c in chain] == ["rollout-mix", "sft-mix", "export-root"]
        assert resp.json()["depth"] == 3

    async def test_chain_stubs_unregistered_ancestor(self, client):
        d = (
            await client.post(
                "/api/v1/datasets",
                json=_body("orphan", SHA_A, lineage={"source_sha256": SHA_B}),
            )
        ).json()
        chain = (await client.get(f"/api/v1/datasets/{d['id']}/lineage")).json()["chain"]
        assert len(chain) == 2
        assert chain[1] == {"sha256": SHA_B, "registered": False}

    async def test_chain_cycle_guard(self, client):
        # A points at B's sha, B points at A's sha — must terminate.
        await client.post("/api/v1/datasets", json=_body("cyc-a", SHA_A, lineage={"source_sha256": SHA_B}))
        b = (
            await client.post(
                "/api/v1/datasets",
                json=_body("cyc-b", SHA_B, lineage={"source_sha256": SHA_A}),
            )
        ).json()
        chain = (await client.get(f"/api/v1/datasets/{b['id']}/lineage")).json()["chain"]
        assert [c["name"] for c in chain if "name" in c] == ["cyc-b", "cyc-a"]
