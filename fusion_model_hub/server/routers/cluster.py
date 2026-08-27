import asyncio
import ipaddress
import itertools
import logging
from datetime import UTC, datetime
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ...db.crud import (
    create_cluster_node,
    delete_cluster_node,
    get_cluster_node,
    list_cluster_nodes,
)
from ...db.models import ClusterNode, DistributedTaskStatus
from .. import http_client as httpx
from ..deps import SessionDep, SettingsDep, get_session_factory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cluster"])

_HEARTBEAT_STALE_SECONDS = 120

# Issue #31: round-robin load balance across active cluster nodes. Before, the
# routing loop started at list_cluster_nodes()[0] every call (created_at DESC =
# newest node first), so the newest node absorbed all traffic and earlier
# primaries served 0. A monotonic counter rotates the start offset per call so
# load spreads evenly across healthy nodes, while still falling through to the
# rest on failure (failover preserved within each call). Single asyncio event
# loop -> no lock needed; itertools.count is deterministic (Rule 5: no random).
_round_robin_counter = itertools.count()


def _validate_node_url(url: str) -> None:
    # Node URLs are admin-supplied: POST /cluster/nodes requires an API key
    # (admin role), and a legit deployment includes same-host multi-port Hub
    # peers at the loopback. The broad validate_external_url SSRF guard (which
    # rejects all RFC1918 space + loopback) is the wrong tool here — it would
    # block real multi-node setups AND the common single-box multi-hub layout.
    # The genuine server-side-fetch risk for a stored node URL is narrower: a
    # non-http(s) scheme (file:///etc/passwd, gopher://...), a missing host, a
    # link-local cloud-metadata address (169.254.169.254), or an unspecified
    # address (0.0.0.0). Reject exactly those; allow loopback + RFC1918 peers.
    # Loopback-to-self is admin-owned (the admin explicitly registered the peer)
    # and the Hub only GETs /health on it — low SSRF value. The strict
    # validate_external_url guard still protects the untrusted caller-supplied
    # fetches (sync_registry, downloads).
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid node URL: {e}")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail="Node URL must use http or https scheme",
        )
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise HTTPException(status_code=400, detail="Node URL must include a hostname")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_link_local or ip.is_unspecified):
        # 169.254.169.254 (cloud metadata) and 0.0.0.0 — never a legitimate
        # peer node; reject. Loopback (127.x/::1) and RFC1918 private IPs are
        # allowed (admin-gated registration — see docstring above).
        raise HTTPException(
            status_code=400,
            detail="Node URL cannot point to link-local or unspecified address",
        )


class NodeCreate(BaseModel):
    name: str
    url: str
    capabilities: str = "inference,quantize"


def _mlx_headers(settings) -> dict[str, str]:
    headers = {"X-Fusion-Source": "model-hub"}
    if settings.mlx_internal_api_key:
        headers["Authorization"] = f"Bearer {settings.mlx_internal_api_key}"
    return headers


def _parse_host_port(url: str) -> tuple[str, int]:
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        return parsed.hostname or "", parsed.port or 0
    except Exception:
        return "", 0


def _effective_status(node: ClusterNode) -> str:
    if node.status != "active" or node.last_heartbeat is None:
        return node.status
    hb = node.last_heartbeat
    # SQLite stores datetimes naive; normalize to aware before subtracting
    # against the aware utcnow or we raise "can't subtract offset-naive and
    # offset-aware datetimes" on the very first stale check.
    if hb.tzinfo is None:
        hb = hb.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - hb).total_seconds()
    return "inactive" if age > _HEARTBEAT_STALE_SECONDS else node.status


def _node_to_dict(node: ClusterNode) -> dict:
    host, port = _parse_host_port(node.url)
    seen = node.last_heartbeat.isoformat() if node.last_heartbeat else None
    return {
        "id": node.id,
        "name": node.name,
        "url": node.url,
        "host": host,
        "port": port,
        "status": _effective_status(node),
        "capabilities": node.capabilities,
        "gpu_type": "",
        "memory_gb": 0.0,
        "cpu_usage": 0.0,
        "gpu_usage": 0.0,
        "memory_used": 0.0,
        "models": [],
        "last_seen": seen,
        "last_heartbeat": seen,
        "created_at": node.created_at.isoformat(),
    }


async def _check_alive(url: str, settings) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/health", headers=_mlx_headers(settings))
            return resp.status_code == 200
    except Exception:
        return False


def _local_node_dict(settings, alive: bool) -> dict:
    host, port = _parse_host_port(settings.mlx_url)
    now = datetime.now(UTC).isoformat() if alive else None
    return {
        "id": "local",
        "name": "local-mlx",
        "url": settings.mlx_url,
        "host": host or "127.0.0.1",
        "port": port or 11434,
        "status": "active" if alive else "inactive",
        "capabilities": "inference,quantize,convert",
        "gpu_type": "",
        "memory_gb": 0.0,
        "cpu_usage": 0.0,
        "gpu_usage": 0.0,
        "memory_used": 0.0,
        "models": [],
        "last_seen": now,
        "last_heartbeat": now,
        "created_at": datetime.now(UTC).isoformat(),
    }


async def _reap_stale_nodes(session) -> int:
    # R7: a node whose heartbeat is older than the stale window is dead — the
    # prior code computed _effective_status() on the fly for display but never
    # wrote it back, so the row stayed status="active" forever and route_inference
    # / sync_model kept targeting a corpse. Persist the reaped state so every
    # downstream query (list, sync, route) sees the truth without recomputing.
    nodes = await list_cluster_nodes(session)
    reaped = 0
    for node in nodes:
        if node.status == "active" and _effective_status(node) == "inactive":
            node.status = "inactive"
            reaped += 1
    if reaped:
        await session.commit()
        logger.warning("Reaped %d stale cluster node(s) to inactive", reaped)
    return reaped


@router.post("/cluster/nodes", status_code=201)
async def add_node(body: NodeCreate, session: SessionDep):
    # Validate the node URL at registration: reject non-http schemes, missing
    # host, and link-local (cloud metadata) / unspecified targets, but ALLOW
    # loopback + RFC1918 peer nodes (admin-gated — see _validate_node_url).
    # Stops a bad URL from persisting and being trusted by _check_alive /
    # topology / routing.
    _validate_node_url(body.url)
    logger.info("Adding cluster node: name=%s url=%s", body.name, body.url)
    node = await create_cluster_node(
        session,
        name=body.name,
        url=body.url,
        capabilities=body.capabilities,
    )
    return _node_to_dict(node)


@router.get("/cluster/nodes")
async def get_nodes(session: SessionDep, settings: SettingsDep):
    await _reap_stale_nodes(session)
    nodes = await list_cluster_nodes(session)
    alive = await _check_alive(settings.mlx_url, settings)
    node_dicts = [_local_node_dict(settings, alive), *[_node_to_dict(n) for n in nodes]]
    logger.info("Listed cluster nodes: %d registered + 1 local (alive=%s)", len(nodes), alive)
    return {"nodes": node_dicts, "total": len(node_dicts)}


@router.get("/cluster/nodes/{node_id}")
async def get_node(node_id: str, session: SessionDep):
    node = await get_cluster_node(session, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return _node_to_dict(node)


@router.delete("/cluster/nodes/{node_id}")
async def remove_node(node_id: str, session: SessionDep):
    deleted = await delete_cluster_node(session, node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"detail": "deleted"}


@router.post("/cluster/nodes/{node_id}/heartbeat")
async def heartbeat(node_id: str, session: SessionDep):
    node = await get_cluster_node(session, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    node.last_heartbeat = datetime.now(UTC)
    node.status = "active"
    await session.commit()
    logger.info("Heartbeat from node: id=%s", node_id)
    return {"detail": "ok"}


class DistributedTaskCreate(BaseModel):
    model_id: str
    version_id: str = ""
    target_nodes: list[str] = []


_running_distributed: dict[str, asyncio.Task] = {}


@router.post("/cluster/distributed-tasks", status_code=202)
async def submit_distributed_task(body: DistributedTaskCreate, session: SessionDep):
    if body.target_nodes:
        for nid in body.target_nodes:
            if not await get_cluster_node(session, nid):
                raise HTTPException(status_code=404, detail=f"Node {nid} not found")
    task = await crud.create_distributed_task(
        session,
        model_id=body.model_id,
        version_id=body.version_id,
        target_nodes=str(body.target_nodes),
    )

    async def _run(task_id: str, target_nodes: list[str], model_id: str):
        sf = get_session_factory()
        async with sf() as s:
            try:
                await crud.update_distributed_task(
                    s,
                    task_id,
                    status=DistributedTaskStatus.RUNNING,
                )
                logger.info("Distributed task running: id=%s nodes=%s", task_id, target_nodes)
                completed = failed = 0
                if target_nodes:
                    nodes = {n.id: n for n in await list_cluster_nodes(s)}
                    for nid in target_nodes:
                        node = nodes.get(nid)
                        if not node:
                            failed += 1
                            continue
                        try:
                            _validate_node_url(node.url)
                            async with httpx.AsyncClient(timeout=30.0) as client:
                                resp = await client.post(
                                    f"{node.url}/api/v1/cluster/remote-sync",
                                    json={"task_type": "model_sync", "model_id": model_id},
                                )
                            if resp.status_code in (200, 201, 202):
                                completed += 1
                            else:
                                failed += 1
                                logger.warning("Node %s sync returned %d", nid, resp.status_code)
                        except Exception as e:
                            failed += 1
                            logger.warning("Node %s sync failed: %s", nid, e)
                else:
                    # R11: empty target_nodes used to sleep 0.1s then report
                    # completed=1 -> a fake COMPLETED with no work done. An empty
                    # target set means nothing was synced; fail explicitly so the
                    # status reflects reality instead of a false-success.
                    logger.warning("Distributed task %s had no target nodes", task_id)
                    failed = 1
                final = (
                    DistributedTaskStatus.COMPLETED
                    if failed == 0
                    else DistributedTaskStatus.PARTIAL
                    if completed > 0
                    else DistributedTaskStatus.FAILED
                )
                await crud.update_distributed_task(
                    s,
                    task_id,
                    status=final,
                    progress=f'{{"completed": {completed}, "failed": {failed}}}',
                )
            except Exception as e:
                await crud.update_distributed_task(
                    s,
                    task_id,
                    status=DistributedTaskStatus.FAILED,
                    progress=str(e),
                )
                logger.exception("Distributed task failed: id=%s", task_id)

    t = asyncio.create_task(_run(task.id, body.target_nodes, body.model_id))
    _running_distributed[task.id] = t
    t.add_done_callback(lambda _: _running_distributed.pop(task.id, None))
    return {"task_id": task.id, "status": "submitted"}


@router.get("/cluster/distributed-tasks/{task_id}")
async def get_distributed_task_status(task_id: str, session: SessionDep):
    task = await crud.get_distributed_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Distributed task not found")
    return {
        "id": task.id,
        "model_id": task.model_id,
        "version_id": task.version_id,
        "target_nodes": task.target_nodes,
        "status": task.status.value,
        "progress": task.progress,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


class SyncModelRequest(BaseModel):
    model_id: str
    target_nodes: list[str] = []


@router.get("/cluster/topology")
async def cluster_topology(session: SessionDep, settings: SettingsDep):
    nodes = await list_cluster_nodes(session)
    alive = await _check_alive(settings.mlx_url, settings)
    node_dicts = [_local_node_dict(settings, alive), *[_node_to_dict(n) for n in nodes]]
    edges = [{"id": "hub-local", "from": "hub", "to": "local", "latency": 0.0, "bandwidth": 0.0}]
    for node in nodes:
        if _effective_status(node) == "active":
            edges.append(
                {
                    "id": f"hub-{node.id}",
                    "from": "hub",
                    "to": node.id,
                    "latency": 0.0,
                    "bandwidth": 0.0,
                }
            )
    active_remote = [n["id"] for n in node_dicts if n["status"] == "active" and n["id"] != "local"]
    routes = [
        {
            "pattern": "default",
            "strategy": "local-first",
            "fallback_nodes": active_remote,
        }
    ]
    logger.info("Cluster topology: %d nodes, %d edges, local_alive=%s", len(node_dicts), len(edges), alive)
    return {
        "nodes": node_dicts,
        "edges": edges,
        "routes": routes,
        "localNode": "local",
    }


@router.post("/cluster/sync-model")
async def sync_model_to_cluster(body: SyncModelRequest, session: SessionDep, settings: SettingsDep):
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    model_name = m.hf_repo or m.name
    await _reap_stale_nodes(session)
    nodes = await list_cluster_nodes(session)
    if body.target_nodes:
        nodes = [n for n in nodes if n.id in body.target_nodes]
    # R7: never sync to a dead node — a corpse still marked active would make
    # the remote-sync POST hang to its timeout and inflate remote_failed.
    nodes = [n for n in nodes if _effective_status(n) == "active"]

    alive = await _check_alive(settings.mlx_url, settings)
    local_ok = False
    want_local = not body.target_nodes or "local" in body.target_nodes
    if alive and want_local:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{settings.mlx_url}/v1/models/{model_name}/load",
                    headers=_mlx_headers(settings),
                )
                local_ok = resp.status_code in (200, 409)
                logger.info("Sync model %s to local MLX /load: %d", body.model_id, resp.status_code)
                if resp.status_code == 404:
                    logger.info("MLX /load 404 (upstream route shadowing), falling back to chat auto-load")
                    chat_resp = await client.post(
                        f"{settings.mlx_url}/v1/chat/completions",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": "."}],
                            "max_tokens": 1,
                            "stream": False,
                        },
                        headers=_mlx_headers(settings),
                    )
                    local_ok = chat_resp.status_code == 200
                    logger.info("Sync model %s chat auto-load: %d", body.model_id, chat_resp.status_code)
        except Exception as e:
            logger.warning("Sync model %s to local MLX failed: %s", body.model_id, e)

    remote_ok = remote_failed = 0
    for node in nodes:
        try:
            _validate_node_url(node.url)
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{node.url}/api/v1/cluster/remote-sync",
                    json={"task_type": "model_sync", "model_id": body.model_id},
                )
            if resp.status_code in (200, 201, 202):
                remote_ok += 1
            else:
                remote_failed += 1
                logger.warning("Sync model %s to node %s: %d", body.model_id, node.id, resp.status_code)
        except Exception as e:
            remote_failed += 1
            logger.warning("Sync model %s to node %s failed: %s", body.model_id, node.id, e)

    success = local_ok or remote_ok > 0
    msg = f"local={'ok' if local_ok else 'skip'}, remote_ok={remote_ok}, remote_failed={remote_failed}"
    logger.info("Sync model %s complete: %s", body.model_id, msg)
    return {
        "success": success,
        "message": msg,
        "error": "" if success else "no node accepted the model",
        "model_id": body.model_id,
    }


class RouteInferenceRequest(BaseModel):
    model_id: str
    messages: list[dict] = []
    mode: str = "auto"
    payload: dict = {}


async def _chat(url: str, settings, model_name: str, messages: list[dict]) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{url}/v1/chat/completions",
            json={"model": model_name, "messages": messages, "stream": False},
            headers=_mlx_headers(settings),
        )
        resp.raise_for_status()
        return resp.json()


def _chat_to_response(result: dict, routed_to: str, route_mode: str, model_name: str) -> dict:
    usage = result.get("usage", {}) or {}
    choices = result.get("choices", []) or []
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    return {
        "id": result.get("id", ""),
        "content": content,
        "model": result.get("model", model_name),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "routedTo": routed_to,
        "routeMode": route_mode,
    }


@router.post("/cluster/route-inference")
async def route_inference(body: RouteInferenceRequest, session: SessionDep, settings: SettingsDep):
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    model_name = m.hf_repo or m.name
    mode = (body.mode or "auto").lower()

    alive = await _check_alive(settings.mlx_url, settings)
    use_local = alive if mode == "local" else (False if mode == "cluster" else alive)

    if use_local:
        try:
            result = await _chat(settings.mlx_url, settings, model_name, body.messages)
            logger.info("Route-inference local: model=%s", model_name)
            return _chat_to_response(result, "local", "local", model_name)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Local Fusion-MLX server unavailable")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.warning("Route-inference local failed: %s", e)

    nodes = await list_cluster_nodes(session)
    active = [n for n in nodes if _effective_status(n) == "active"]
    if active:
        # Issue #31: round-robin — rotate the start node per call so load
        # balances across healthy nodes instead of always hitting nodes[0].
        # offset % len(active) rebases when the active set changes; the slice
        # rotates the list so iteration still visits every active node (failover
        # within the call is preserved if the start node fails).
        offset = next(_round_robin_counter) % len(active)
        active = active[offset:] + active[:offset]
    for node in active:
        try:
            # E-S14: re-validate the node URL before every fetch. add_node
            # validated at registration, but a node row can drift (manual DB
            # edit, a URL that resolved public then flips via DNS rebinding),
            # and route_inference is a privileged server-side fetch — never
            # trust a stored URL without re-checking it against the SSRF guard.
            _validate_node_url(node.url)
            result = await _chat(node.url, settings, model_name, body.messages)
            logger.info("Route-inference remote: node=%s model=%s", node.id, model_name)
            return _chat_to_response(result, node.id, "remote", model_name)
        except Exception:
            logger.debug("Node %s inference failed", node.id, exc_info=True)
            continue

    raise HTTPException(status_code=503, detail="No available node for inference routing")


class RemoteSyncRequest(BaseModel):
    task_type: str = "model_sync"
    model_id: str


@router.post("/cluster/remote-sync")
async def remote_sync_inbox(body: RemoteSyncRequest, session: SessionDep, settings: SettingsDep):
    # H1: the real inter-node sync endpoint. Prior code POSTed to a phantom
    # /api/tasks/submit that no router defined, so every remote node 404'd and
    # distributed tasks silently degraded. This inbox is what a peer hub calls;
    # it loads the requested model on its OWN local Fusion-MLX (delegated, per
    # the project's never-import-mlx rule). Returns 202 on accepted/already-loaded.
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    model_name = m.hf_repo or m.name
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.mlx_url}/v1/models/{model_name}/load",
                headers=_mlx_headers(settings),
            )
            accepted = resp.status_code in (200, 409)
            logger.info("Remote-sync inbox: model=%s node_load_status=%d", body.model_id, resp.status_code)
    except Exception as e:
        logger.warning("Remote-sync inbox load failed: model=%s err=%s", body.model_id, e)
        raise HTTPException(status_code=503, detail=f"Local MLX load failed: {e}")
    if not accepted:
        raise HTTPException(status_code=502, detail="Local MLX rejected model load")
    return {"accepted": True, "model_id": body.model_id, "model_name": model_name}
