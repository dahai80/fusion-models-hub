import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import SessionDep
from ...db.crud import (
    create_cluster_node,
    delete_cluster_node,
    get_cluster_node,
    list_cluster_nodes,
)
from ...db.models import ClusterNode

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
    from datetime import datetime, timezone
    node.last_heartbeat = datetime.now(timezone.utc)
    node.status = "active"
    await session.commit()
    logger.info("Heartbeat from node: id=%s", node_id)
    return {"detail": "ok"}
