import asyncio
import logging
from datetime import UTC

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
from ..deps import SessionDep, get_session_factory

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

    async def _run_distributed(task_id: str):
        sf = get_session_factory()
        async with sf() as s:
            try:
                await crud.update_distributed_task(
                    s, task_id, status=DistributedTaskStatus.RUNNING,
                )
                logger.info("Distributed task running: id=%s", task_id)
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

    t = asyncio.create_task(_run_distributed(task.id))
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
