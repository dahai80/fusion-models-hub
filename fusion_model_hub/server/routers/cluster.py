import asyncio
import logging
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx
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
from ..deps import SessionDep, SettingsDep, get_session_factory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cluster"])

_HEARTBEAT_STALE_SECONDS = 120


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
    age = (datetime.now(UTC) - node.last_heartbeat).total_seconds()
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


@router.post("/cluster/nodes", status_code=201)
async def add_node(body: NodeCreate, session: SessionDep):
    logger.info("Adding cluster node: name=%s url=%s", body.name, body.url)
    node = await create_cluster_node(
        session, name=body.name, url=body.url, capabilities=body.capabilities,
    )
    return _node_to_dict(node)


@router.get("/cluster/nodes")
async def get_nodes(session: SessionDep, settings: SettingsDep):
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
        session, model_id=body.model_id, version_id=body.version_id,
        target_nodes=str(body.target_nodes),
    )

    async def _run(task_id: str, target_nodes: list[str], model_id: str):
        sf = get_session_factory()
        async with sf() as s:
            try:
                await crud.update_distributed_task(
                    s, task_id, status=DistributedTaskStatus.RUNNING,
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
                            async with httpx.AsyncClient(timeout=30.0) as client:
                                resp = await client.post(
                                    f"{node.url}/api/tasks/submit",
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
                    await asyncio.sleep(0.1)
                    completed = 1
                final = (
                    DistributedTaskStatus.COMPLETED if failed == 0
                    else DistributedTaskStatus.PARTIAL if completed > 0
                    else DistributedTaskStatus.FAILED
                )
                await crud.update_distributed_task(
                    s, task_id, status=final,
                    progress=f'{{"completed": {completed}, "failed": {failed}}}',
                )
            except Exception as e:
                await crud.update_distributed_task(
                    s, task_id, status=DistributedTaskStatus.FAILED, progress=str(e),
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
            edges.append({
                "id": f"hub-{node.id}", "from": "hub", "to": node.id,
                "latency": 0.0, "bandwidth": 0.0,
            })
    active_remote = [n["id"] for n in node_dicts if n["status"] == "active" and n["id"] != "local"]
    routes = [{
        "pattern": "default",
        "strategy": "local-first",
        "fallback_nodes": active_remote,
    }]
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
    nodes = await list_cluster_nodes(session)
    if body.target_nodes:
        nodes = [n for n in nodes if n.id in body.target_nodes]

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
                        json={"model": model_name, "messages": [{"role": "user", "content": "."}], "max_tokens": 1, "stream": False},
                        headers=_mlx_headers(settings),
                    )
                    local_ok = chat_resp.status_code == 200
                    logger.info("Sync model %s chat auto-load: %d", body.model_id, chat_resp.status_code)
        except Exception as e:
            logger.warning("Sync model %s to local MLX failed: %s", body.model_id, e)

    remote_ok = remote_failed = 0
    for node in nodes:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{node.url}/api/tasks/submit",
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
    for node in nodes:
        if _effective_status(node) != "active":
            continue
        try:
            result = await _chat(node.url, settings, model_name, body.messages)
            logger.info("Route-inference remote: node=%s model=%s", node.id, model_name)
            return _chat_to_response(result, node.id, "remote", model_name)
        except Exception:
            logger.debug("Node %s inference failed", node.id, exc_info=True)
            continue

    raise HTTPException(status_code=503, detail="No available node for inference routing")
