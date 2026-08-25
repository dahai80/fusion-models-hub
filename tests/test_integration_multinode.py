"""End-to-end multi-node smoke test: two real Hub instances over real TCP.

NOT part of the default pytest run. The default cluster suite runs every test
against a single in-process ASGITransport app and mocks httpx for any outbound
call — so the cross-node path (Hub A registers Hub B, dispatches a distributed
task that POSTs to Hub B's /cluster/remote-sync) is never exercised over a real
socket. This module launches two real uvicorn servers on loopback ports, each
with its own Settings/engine/data_dir, and drives the inter-node flow through
httpx.AsyncClient(base_url=real-port) — exactly what production does.

Skips automatically when the loopback ports can't be bound (taken) or uvicorn
isn't importable, so the default suite stays green. Auth OFF. MLX NOT required
— the distributed task asserts the inter-node HTTP plumbing reaches Hub B and
records a terminal status; COMPLETED requires MLX (covered separately by the
MLX E2E suite, Step2). Here we prove Hub A talks to Hub B over a real socket.
"""

import asyncio
import contextlib
import os
import shutil
import socket
import time

import httpx
import pytest

PORT_A = int(os.environ.get("FMH_INT_NODE_PORT_A", "11544"))
PORT_B = int(os.environ.get("FMH_INT_NODE_PORT_B", "11545"))


def _uvicorn_available() -> bool:
    try:
        import uvicorn  # noqa: F401

        return True
    except ImportError:
        return False


def _port_free(port: int) -> bool:
    with contextlib.suppress(OSError):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    return False


_UVICORN_OK = _uvicorn_available()
_PORTS_OK = _port_free(PORT_A) and _port_free(PORT_B)

_SKIP_REASON = (
    "multi-node smoke test skipped: "
    f"uvicorn={_UVICORN_OK} ports_free={_PORTS_OK} "
    f"(ports {PORT_A}/{PORT_B} taken? set FMH_INT_NODE_PORT_A/B)"
)
requires_multinode = pytest.mark.skipif(
    not (_UVICORN_OK and _PORTS_OK),
    reason=_SKIP_REASON,
)


class _Hub:
    # One real uvicorn server bound to a loopback port, with its own Settings,
    # engine, data_dir. Started/stopped within the test event loop so the whole
    # multi-node flow runs over real TCP sockets, not ASGITransport.

    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.data_dir = f"/tmp/fmh_int_node_{name}_{int(time.time())}"
        self.engine = None
        self.server = None
        self.serve_task = None

    async def start(self):
        import uvicorn

        from fusion_model_hub.db.database import get_engine, init_db
        from fusion_model_hub.server.app import create_app
        from fusion_model_hub.server.auth import set_auth_enabled
        from fusion_model_hub.server.config import Settings
        from fusion_model_hub.server.deps import init_deps

        os.makedirs(self.data_dir, exist_ok=True)
        self.settings = Settings(
            host="127.0.0.1",
            port=self.port,
            data_dir=self.data_dir,
            db_url=f"sqlite+aiosqlite:///{self.data_dir}/hub.db",
            log_level="WARNING",
            # Auth off via Settings so create_app's lifespan
            # (set_auth_enabled(settings.auth_enabled)) keeps it off for both
            # hubs — a manual set_auth_enabled(False) here would be overwritten
            # by each hub's own startup.
            auth_enabled=False,
            # Point MLX at a dead port so no accidental MLX load interferes;
            # the smoke test asserts HTTP plumbing, not inference.
            mlx_url="http://127.0.0.1:1",
        )
        set_auth_enabled(False)
        self.engine = get_engine(self.settings.db_url)
        await init_db(self.engine)
        init_deps(self.settings, self.engine)
        app = create_app(self.settings)
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port,
            log_level="warning", access_log=False, lifespan="on",
        )
        self.server = uvicorn.Server(config)
        # Run serve() in the background; should_exit stops it in teardown.
        self.serve_task = asyncio.create_task(self.server.serve())
        # Wait until the socket is accepting connections.
        for _ in range(100):
            with contextlib.suppress(OSError):
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError(f"Hub {self.name} did not bind port {self.port}")

    async def stop(self):
        if self.server is not None:
            self.server.should_exit = True
        if self.serve_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(self.serve_task, timeout=10)
        if self.engine is not None:
            await self.engine.dispose()
        shutil.rmtree(self.data_dir, ignore_errors=True)


@pytest.fixture
async def two_hubs():
    hub_a = _Hub("a", PORT_A)
    hub_b = _Hub("b", PORT_B)
    await hub_a.start()
    await hub_b.start()
    try:
        yield hub_a, hub_b
    finally:
        await hub_a.stop()
        await hub_b.stop()


class TestMultiNodeCluster:
    # Hub A registers Hub B as a cluster node, then drives the inter-node flow
    # over real TCP. Proves the cross-node path the default mocked suite skips.

    @pytest.mark.asyncio
    @requires_multinode
    async def test_hub_b_reachable_over_real_socket(self, two_hubs):
        hub_a, hub_b = two_hubs
        async with httpx.AsyncClient(base_url=hub_b.url, timeout=5.0) as b_client:
            resp = await b_client.get("/api/v1/system/health")
            assert resp.status_code == 200, "Hub B server not listening on real socket"
            assert resp.json()["status"] in ("healthy", "degraded")

    @pytest.mark.asyncio
    @requires_multinode
    async def test_register_and_list_node_cross_hub(self, two_hubs):
        hub_a, hub_b = two_hubs
        async with httpx.AsyncClient(base_url=hub_a.url, timeout=10.0) as a_client:
            reg = await a_client.post("/api/v1/cluster/nodes", json={
                "name": "hub-b",
                "url": hub_b.url,
                "capabilities": "inference,quantize",
            })
            assert reg.status_code == 201, reg.text
            node_id = reg.json()["id"]

            # List on Hub A must include the local node + the registered Hub B.
            lst = await a_client.get("/api/v1/cluster/nodes")
            assert lst.status_code == 200
            ids = [n["id"] for n in lst.json()["nodes"]]
            assert "local" in ids
            assert node_id in ids
            assert lst.json()["total"] >= 2

            # Topology must surface an edge hub->node_id once Hub B is active.
            topo = await a_client.get("/api/v1/cluster/topology")
            assert topo.status_code == 200
            edge_targets = [e["to"] for e in topo.json()["edges"]]
            assert node_id in edge_targets, "active node missing from topology edges"

    @pytest.mark.asyncio
    @requires_multinode
    async def test_distributed_task_reaches_remote_hub(self, two_hubs):
        # Hub A submits a distributed task targeting Hub B. The _run loop POSTs
        # to Hub B's /cluster/remote-sync over a real socket. Hub B has no MLX
        # (mlx_url=dead port), so its inbox returns 503 -> Hub A records FAILED.
        # A terminal FAILED status (not a hung RUNNING) proves the POST reached
        # a real Hub B server and got a real response back — the HTTP plumbing
        # the default mocked suite never exercises. COMPLETED needs MLX (Step2).
        hub_a, hub_b = two_hubs
        async with httpx.AsyncClient(base_url=hub_a.url, timeout=15.0) as a_client:
            model = await a_client.post("/api/v1/models", json={
                "name": "multi-node-model",
                "model_type": "llm",
            })
            assert model.status_code == 201, model.text
            model_id = model.json()["id"]

            reg = await a_client.post("/api/v1/cluster/nodes", json={
                "name": "hub-b",
                "url": hub_b.url,
            })
            assert reg.status_code == 201, reg.text
            node_id = reg.json()["id"]

            submit = await a_client.post("/api/v1/cluster/distributed-tasks", json={
                "model_id": model_id,
                "target_nodes": [node_id],
            })
            assert submit.status_code == 202, submit.text
            task_id = submit.json()["task_id"]

            # Poll until the async _run loop lands a terminal status.
            status = None
            for _ in range(60):
                got = await a_client.get(f"/api/v1/cluster/distributed-tasks/{task_id}")
                assert got.status_code == 200, got.text
                status = got.json()["status"]
                if status in ("completed", "failed", "partial"):
                    break
                await asyncio.sleep(0.2)
            assert status in ("failed", "partial"), (
                f"distributed task did not reach terminal status over real socket: {status}"
            )
            # The task record must carry the target node we dispatched to.
            assert node_id in got.json()["target_nodes"]
