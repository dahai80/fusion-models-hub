import asyncio
import logging
from datetime import UTC

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


class NodeCreate(BaseModel):
    name: str
    url: str
    capabilities: str = "inference,quantize"


class NodeUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    status: str | None = None
    capabilities: str | None = None


def _node_to_dict(node: ClusterNode) -> dict:
    return {
        "id": node.id,
        "name": node.name,
        "url": node.url,
        "status": node.status,
        "capabilities": node.capabilities,
        "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None,
        "created_at": node.created_at.isoformat(),
    }


@router.post("/cluster/nodes", status_code=201)
async def add_node(body: NodeCreate, session: SessionDep):
    logger.info("Adding cluster node: name=%s url=%s", body.name, body.url)
    node = await create_cluster_node(
        session,
        name=body.name,
        url=body.url,
        capabilities=body.capabilities,
    )
    return _node_to_dict(node)


@router.get("/cluster/nodes")
async def get_nodes(session: SessionDep):
    nodes = await list_cluster_nodes(session)
    return [_node_to_dict(n) for n in nodes]


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
    from datetime import datetime
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
            node = await get_cluster_node(session, nid)
            if not node:
                raise HTTPException(status_code=404, detail=f"Node {nid} not found")
    task = await crud.create_distributed_task(
        session,
        model_id=body.model_id,
        version_id=body.version_id,
        target_nodes=str(body.target_nodes),
    )

    async def _run_distributed(task_id: str, target_nodes: list[str], model_id: str):
        sf = get_session_factory()
        settings = None
        try:
            from ..deps import get_settings
            settings = get_settings()
        except Exception:
            settings = None
        async with sf() as s:
            try:
                await crud.update_distributed_task(
                    s, task_id, status=DistributedTaskStatus.RUNNING,
                )
                logger.info("Distributed task running: id=%s nodes=%s", task_id, target_nodes)
                if settings and target_nodes:
                    nodes = await list_cluster_nodes(s)
                    node_map = {n.id: n for n in nodes}
                    completed = 0
                    failed = 0
                    for nid in target_nodes:
                        node = node_map.get(nid)
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
                    final_status = DistributedTaskStatus.COMPLETED if failed == 0 else (
                        DistributedTaskStatus.PARTIAL if completed > 0 else DistributedTaskStatus.FAILED
                    )
                    await crud.update_distributed_task(
                        s, task_id, status=final_status,
                        progress=f'{{"completed": {completed}, "failed": {failed}}}',
                    )
                else:
                    await asyncio.sleep(0.1)
                    await crud.update_distributed_task(
                        s, task_id, status=DistributedTaskStatus.COMPLETED,
                        progress="{}",
                    )
            except Exception as e:
                await crud.update_distributed_task(
                    s, task_id, status=DistributedTaskStatus.FAILED,
                    progress=str(e),
                )
                logger.exception("Distributed task failed: id=%s", task_id)

    t = asyncio.create_task(_run_distributed(task.id, body.target_nodes, body.model_id))
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
    nodes_data = [_node_to_dict(n) for n in nodes]
    edges = []
    for node in nodes:
        if node.status == "active":
            edges.append({"source": "hub", "target": node.id, "type": "management"})
    routes = []
    active_nodes = [n for n in nodes if n.status == "active"]
    if active_nodes:
        routes.append({
            "pattern": "default",
            "strategy": "local-first",
            "fallback_nodes": [n.id for n in active_nodes],
        })
    return {
        "nodes": [*[
            {
                "id": "hub",
                "name": "model-hub",
                "status": "active",
                "address": f"http://{settings.host}:{settings.port}",
            }
        ], *nodes_data],
        "edges": edges,
        "routes": routes,
    }


@router.post("/cluster/sync-model")
async def sync_model_to_cluster(body: SyncModelRequest, session: SessionDep, settings: SettingsDep):
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    nodes = await list_cluster_nodes(session)
    if body.target_nodes:
        nodes = [n for n in nodes if n.id in body.target_nodes]
    results = []
    for node in nodes:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{node.url}/api/tasks/submit",
                    json={"task_type": "model_sync", "model_id": body.model_id},
                )
                results.append({"node_id": node.id, "status": resp.status_code})
                logger.info("Sync model %s to node %s: %d", body.model_id, node.id, resp.status_code)
        except Exception as e:
            results.append({"node_id": node.id, "status": "error", "detail": str(e)})
            logger.warning("Sync model %s to node %s failed: %s", body.model_id, node.id, e)
    return {"model_id": body.model_id, "results": results}


class RouteInferenceRequest(BaseModel):
    model_id: str
    payload: dict = {}


@router.post("/cluster/route-inference")
async def route_inference(body: RouteInferenceRequest, session: SessionDep, settings: SettingsDep):
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            stats_resp = await client.get(f"{settings.mlx_url}/metrics")
            local_ok = stats_resp.status_code == 200
    except Exception:
        local_ok = False

    if local_ok:
        return {"route": "local", "model_id": body.model_id}

    nodes = await list_cluster_nodes(session)
    active_nodes = [n for n in nodes if n.status == "active"]
    for node in active_nodes:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{node.url}/health")
                if resp.status_code == 200:
                    logger.info("Routing inference to node %s", node.id)
                    return {"route": "remote", "node_id": node.id, "node_url": node.url, "model_id": body.model_id}
        except Exception:
            logger.debug("Node %s health check failed", node.id, exc_info=True)
            continue
    raise HTTPException(status_code=503, detail="No available node for inference routing")
