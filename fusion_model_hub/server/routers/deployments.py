import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ...db.models import DeploymentStatus
from ..deps import SessionDep, SettingsDep
from .cluster import _effective_status, get_cluster_node
from .inference import _mlx_headers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deployments", tags=["deployments"])


async def _resolve_node_url(session, node_id: str, settings) -> str:
    # H3: map a deployment's node_id back to the MLX URL the model was
    # placed on. "local"/"" → local MLX; otherwise look up the ClusterNode.
    # Falls back to the local MLX URL if the node row is gone (deleted node),
    # logged — better than a hard failure on unload.
    node_id = (node_id or "local").strip()
    if not node_id or node_id == "local":
        return settings.mlx_url
    node = await get_cluster_node(session, node_id)
    if not node:
        logger.warning("Deployment node %s no longer registered; falling back to local MLX", node_id)
        return settings.mlx_url
    return node.url


async def _load_model_via_mlx(settings, model_name: str, node_url: str | None = None) -> bool:
    # H3: node_url lets a deployment load its model on a specific cluster
    # node's MLX instead of always the local one. None/empty falls back to
    # settings.mlx_url (legacy behavior).
    base_url = (node_url or settings.mlx_url).rstrip("/")
    headers = _mlx_headers(settings)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/v1/models/{model_name}/load", headers=headers,
            )
            if resp.status_code in (200, 409):
                return True
            if resp.status_code == 404:
                logger.info(
                    "MLX /load 404 for %s (upstream route shadowing), falling back to chat auto-load",
                    model_name,
                )
                chat_resp = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "."}],
                        "max_tokens": 1,
                        "stream": False,
                    },
                    headers=headers,
                )
                return chat_resp.status_code == 200
            logger.warning("MLX load returned %d for %s", resp.status_code, model_name)
    except httpx.ConnectError:
        logger.warning("Fusion-MLX server unavailable for load of %s", model_name)
    except Exception as e:
        logger.warning("MLX load call failed for %s: %s", model_name, e)
    return False


async def _fetch_mlx_inference_metrics(settings, model_name: str, node_url: str | None = None) -> dict:
    # fusion-mlx exposes inference throughput/error counters via
    # GET /v1/metrics/json (upstream PR dahai80/fusion-mlx#541), mgmt-gated,
    # returning ServerMetrics.to_dict(): total_requests / successful_requests /
    # failed_requests / active_requests / avg_generation_tps / uptime_seconds +
    # per-model model_stats. Derive the four studio deployment-metrics fields
    # that were previously always null:
    #   requestsPerSecond   = total_requests / uptime_seconds
    #   errorRate           = failed_requests / total_requests   (fraction 0..1)
    #   activeConnections   = active_requests
    #   tokensPerSecond     = avg_generation_tps
    # Tolerates 404 (endpoint absent until #541 merges) by returning {} so the
    # caller keeps the keys null — shape stays stable.
    # H3: node_url queries the node the deployment is actually placed on.
    base_url = (node_url or settings.mlx_url).rstrip("/")
    headers = _mlx_headers(settings)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{base_url}/v1/metrics/json", headers=headers,
            )
            if resp.status_code == 404:
                logger.info(
                    "MLX /v1/metrics/json 404 for %s (PR #541 not merged); "
                    "deployment metrics 4 fields stay null",
                    model_name,
                )
                return {}
            if resp.status_code != 200:
                logger.warning(
                    "MLX /v1/metrics/json returned %d for %s", resp.status_code, model_name,
                )
                return {}
            data = resp.json()
            if not isinstance(data, dict):
                return {}
            stats = data.get("model_stats", {}).get(model_name) if isinstance(
                data.get("model_stats"), dict
            ) else None
            source = stats or data
            total = float(source.get("total_requests") or 0)
            failed = float(source.get("failed_requests") or 0)
            uptime = float(source.get("uptime_seconds") or 0)
            active = source.get("active_requests")
            gen_tps = source.get("avg_generation_tps")
            rps = round(total / uptime, 3) if uptime > 0 else None
            error_rate = round(failed / total, 4) if total > 0 else 0.0
            tokens_per_sec = round(float(gen_tps), 2) if gen_tps is not None else None
            active_conn = int(active) if active is not None else None
            logger.info(
                "MLX inference metrics for %s: rps=%s err=%s active=%s tps=%s",
                model_name, rps, error_rate, active_conn, tokens_per_sec,
            )
            return {
                "requestsPerSecond": rps,
                "errorRate": error_rate,
                "activeConnections": active_conn,
                "tokensPerSecond": tokens_per_sec,
            }
    except httpx.ConnectError:
        logger.warning("Fusion-MLX server unavailable for metrics of %s", model_name)
    except Exception as e:
        logger.warning("MLX metrics call failed for %s: %s", model_name, e)
    return {}



class DeploymentCreate(BaseModel):
    model_id: str = Field(..., min_length=1)
    name: str = ""
    version_id: str = ""
    replicas: int = Field(1, ge=1, le=100)
    # H3: explicit placement. "local" (default) → hub's own Fusion-MLX;
    # a ClusterNode.id → load the model on that node's MLX URL. Empty keeps
    # the legacy local-only behavior so existing callers stay compatible.
    node_id: str = Field("local", min_length=0, max_length=16)
    # fusion-studio aliases
    scale: int | None = Field(None, ge=1, le=100)
    strategy: str | None = None
    canary_percent: int | None = Field(None, ge=0, le=100)
    auto_start: bool | None = None

    @property
    def effective_replicas(self) -> int:
        return self.scale if self.scale is not None else self.replicas

    def effective_name(self, model_name: str = "") -> str:
        return self.name or f"deployment-{model_name}"


class DeploymentUpdate(BaseModel):
    replicas: int | None = None
    version_id: str | None = None
    status: str | None = None


class DeploymentOut(BaseModel):
    id: str
    tenant_id: str
    model_id: str
    version_id: str
    name: str
    replicas: int
    node_id: str
    status: str
    gray_enabled: bool
    gray_version_id: str
    gray_traffic_ratio: int
    model_config = {"from_attributes": True}


def _deployment_to_response(d, model_name: str = "") -> dict:
    # snake_case = hub canonical (tests assert these)
    # camelCase keys = fusion-studio HubDeployment decode (plain JSONDecoder, default keys)
    status_val = d.status.value if hasattr(d.status, "value") else str(d.status)
    created = d.created_at.isoformat() if getattr(d, "created_at", None) else None
    updated = d.updated_at.isoformat() if getattr(d, "updated_at", None) else None
    return {
        "id": d.id,
        "tenant_id": d.tenant_id,
        "model_id": d.model_id,
        "version_id": d.version_id,
        "name": d.name,
        "replicas": d.replicas,
        "node_id": d.node_id,
        "status": status_val,
        "gray_enabled": d.gray_enabled,
        "gray_version_id": d.gray_version_id,
        "gray_traffic_ratio": d.gray_traffic_ratio,
        # studio camelCase mirror
        "modelId": d.model_id,
        "modelName": model_name or d.name,
        "scale": d.replicas,
        "canaryPercent": d.gray_traffic_ratio if d.gray_enabled else 0,
        "strategy": "gray" if d.gray_enabled else "rolling",
        "createdAt": created,
        "updatedAt": updated,
    }


@router.post("", status_code=201)
async def create_deployment(body: DeploymentCreate, session: SessionDep, request: Request, settings: SettingsDep):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    model_name = m.hf_repo or m.name

    # H3: resolve placement. node_id "local"/"" → hub's own MLX (legacy
    # behavior). Any other id must reference a registered, active ClusterNode;
    # load the model on THAT node's URL, not the local one. Without this the
    # deployment always fell on the local MLX regardless of replicas/node.
    node_id = (body.node_id or "local").strip()
    node_url = settings.mlx_url
    if node_id and node_id != "local":
        node = await get_cluster_node(session, node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Cluster node {node_id} not found")
        if _effective_status(node) != "active":
            raise HTTPException(status_code=409, detail=f"Cluster node {node_id} is not active")
        node_url = node.url
        logger.info("Deployment placed on remote node: id=%s url=%s", node_id, node_url)

    mlx_loaded = await _load_model_via_mlx(settings, model_name, node_url)
    dep_name = body.effective_name(model_name)
    d = await crud.create_deployment(
        session, model_id=body.model_id, name=dep_name,
        tenant_id=tenant_id, version_id=body.version_id, replicas=body.effective_replicas,
        node_id=node_id,
    )
    initial_status = DeploymentStatus.RUNNING if mlx_loaded else DeploymentStatus.PENDING
    await crud.update_deployment(session, d.id, status=initial_status)
    d = await crud.get_deployment(session, d.id)
    logger.info("Deployment created: id=%s model=%s replicas=%d node=%s", d.id, model_name, d.replicas, d.node_id)
    return _deployment_to_response(d, model_name)


@router.get("")
async def list_deployments(
    session: SessionDep,
    request: Request,
    model_id: str = "",
    status: str = "",
):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    items = await crud.list_deployments(session, model_id=model_id, status=status, tenant_id=tenant_id)
    name_map = {}
    for it in items:
        if it.model_id not in name_map:
            m = await crud.get_model(session, it.model_id)
            name_map[it.model_id] = (m.hf_repo or m.name) if m else it.name
    resp_items = [_deployment_to_response(it, name_map.get(it.model_id, "")) for it in items]
    logger.info("Listed deployments: %d (tenant=%s)", len(resp_items), tenant_id)
    return {"deployments": resp_items, "total": len(resp_items)}


@router.get("/{deployment_id}")
async def get_deployment(deployment_id: str, session: SessionDep):
    d = await crud.get_deployment(session, deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    m = await crud.get_model(session, d.model_id)
    model_name = (m.hf_repo or m.name) if m else ""
    return _deployment_to_response(d, model_name)


@router.patch("/{deployment_id}")
async def update_deployment(deployment_id: str, body: DeploymentUpdate, session: SessionDep):
    fields = {}
    if body.replicas is not None:
        fields["replicas"] = body.replicas
    if body.version_id is not None:
        fields["version_id"] = body.version_id
    if body.status is not None:
        try:
            DeploymentStatus(body.status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")
        fields["status"] = DeploymentStatus(body.status)
    d = await crud.update_deployment(session, deployment_id, **fields)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    m = await crud.get_model(session, d.model_id)
    model_name = (m.hf_repo or m.name) if m else ""
    return _deployment_to_response(d, model_name)


@router.delete("/{deployment_id}")
async def delete_deployment(deployment_id: str, session: SessionDep, settings: SettingsDep):
    d = await crud.get_deployment(session, deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    m = await crud.get_model(session, d.model_id)
    if m:
        model_name = m.hf_repo or m.name
        # H3: unload on the node the model was actually placed on, not always
        # the local MLX.
        node_url = await _resolve_node_url(session, d.node_id, settings)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{node_url}/v1/models/{model_name}/unload",
                    headers=_mlx_headers(settings),
                )
        except Exception as e:
            logger.warning("MLX unload call failed: %s", e)
    await crud.delete_deployment(session, deployment_id)
    return {"detail": "deleted"}


class GrayReleaseRequest(BaseModel):
    gray_version_id: str = Field("", min_length=0)
    gray_traffic_ratio: int = Field(10, ge=1, le=100)


@router.post("/{deployment_id}/stop")
async def stop_deployment(deployment_id: str, session: SessionDep, settings: SettingsDep):
    d = await crud.get_deployment(session, deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    m = await crud.get_model(session, d.model_id)
    model_name = ""
    if m:
        model_name = m.hf_repo or m.name
        # H3: unload on the node the model was actually placed on.
        node_url = await _resolve_node_url(session, d.node_id, settings)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{node_url}/v1/models/{model_name}/unload",
                    headers=_mlx_headers(settings),
                )
        except Exception as e:
            logger.warning("MLX unload on stop failed: %s", e)
    d = await crud.update_deployment(session, deployment_id, status=DeploymentStatus.STOPPED)
    logger.info("Deployment stopped: id=%s", deployment_id)
    return _deployment_to_response(d, model_name)


@router.post("/{deployment_id}/gray")
async def enable_gray_release(deployment_id: str, body: GrayReleaseRequest, session: SessionDep):
    d = await crud.update_deployment(
        session, deployment_id,
        gray_enabled=True,
        gray_version_id=body.gray_version_id,
        gray_traffic_ratio=body.gray_traffic_ratio,
    )
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    logger.info(
        "Gray release enabled: deployment=%s gray_ver=%s ratio=%d",
        deployment_id, body.gray_version_id, body.gray_traffic_ratio,
    )
    m = await crud.get_model(session, d.model_id)
    return _deployment_to_response(d, (m.hf_repo or m.name) if m else "")


@router.delete("/{deployment_id}/gray")
async def disable_gray_release(deployment_id: str, session: SessionDep):
    d = await crud.update_deployment(
        session, deployment_id,
        gray_enabled=False,
        gray_version_id="",
        gray_traffic_ratio=0,
    )
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    logger.info("Gray release disabled: deployment=%s", deployment_id)
    m = await crud.get_model(session, d.model_id)
    return _deployment_to_response(d, (m.hf_repo or m.name) if m else "")


class ScaleRequest(BaseModel):
    replicas: int | None = Field(None, ge=1, le=100)
    scale: int | None = Field(None, ge=1, le=100)

    @property
    def effective_replicas(self) -> int:
        val = self.replicas if self.replicas is not None else self.scale
        if val is None:
            raise ValueError("replicas or scale required")
        return val


@router.post("/{deployment_id}/scale")
async def scale_deployment(deployment_id: str, body: ScaleRequest, session: SessionDep, settings: SettingsDep):
    d = await crud.get_deployment(session, deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    try:
        replicas = body.effective_replicas
    except ValueError:
        raise HTTPException(status_code=400, detail="replicas or scale required")
    d = await crud.update_deployment(session, deployment_id, replicas=replicas)
    m_name = ""
    if replicas > 0 and d.status == DeploymentStatus.RUNNING:
        m = await crud.get_model(session, d.model_id)
        if m:
            m_name = m.hf_repo or m.name
            await _load_model_via_mlx(settings, m_name)
    logger.info("Deployment scaled: id=%s replicas=%d", deployment_id, replicas)
    return _deployment_to_response(d, m_name)


@router.get("/{deployment_id}/metrics")
async def get_deployment_metrics(deployment_id: str, session: SessionDep, settings: SettingsDep):
    d = await crud.get_deployment(session, deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    m = await crud.get_model(session, d.model_id)
    model_name = m.hf_repo or m.name if m else ""
    # H3: query the node the deployment is placed on, not always local MLX.
    node_url = await _resolve_node_url(session, d.node_id, settings)
    mlx_metrics = {}
    inference_fields = {}
    if model_name and d.status == DeploymentStatus.RUNNING:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{node_url}/v1/models/status",
                    headers=_mlx_headers(settings),
                )
                if resp.status_code == 200:
                    status_data = resp.json()
                    if isinstance(status_data, dict):
                        mlx_metrics = status_data.get(model_name, status_data)
        except Exception as e:
            logger.warning("Failed to fetch MLX metrics: %s", e)
        # /v1/models/status gives load state; /v1/metrics/json (PR #541) gives
        # inference throughput/error counters. Derive the 4 studio live fields
        # from the latter. Falls back to null if the endpoint is absent (404).
        inference_fields = await _fetch_mlx_inference_metrics(settings, model_name, node_url)
    version_metrics = {}
    if d.version_id:
        v = await crud.get_version(session, d.version_id)
        if v:
            version_metrics = {
                "inference_latency": v.inference_latency,
                "throughput": v.throughput,
                "memory_usage": v.memory_usage,
                "benchmark_score": v.benchmark_score,
            }
    # studio HubDeploymentMetricsResponse mirror (all optional):
    #   deploymentId / requestsPerSecond / avgLatencyMs / errorRate /
    #   tokensPerSecond / activeConnections.
    # avgLatencyMs <- version.inference_latency (ms); requestsPerSecond /
    # errorRate / activeConnections / tokensPerSecond <- MLX /v1/metrics/json
    # (null when endpoint absent or deployment not RUNNING). Emitted always so
    # the shape is stable + testable.
    avg_latency = float(version_metrics.get("inference_latency") or 0.0)
    ver_tps = float(version_metrics.get("throughput") or 0.0)
    live_tps = inference_fields.get("tokensPerSecond")
    return {
        "deployment_id": deployment_id,
        "status": d.status.value,
        "replicas": d.replicas,
        "gray_enabled": d.gray_enabled,
        "mlx_metrics": mlx_metrics,
        "version_metrics": version_metrics,
        "deploymentId": deployment_id,
        "requestsPerSecond": inference_fields.get("requestsPerSecond"),
        "avgLatencyMs": round(avg_latency, 2) if avg_latency > 0 else None,
        "errorRate": inference_fields.get("errorRate"),
        "tokensPerSecond": live_tps if live_tps is not None else (
            round(ver_tps, 2) if ver_tps > 0 else None
        ),
        "activeConnections": inference_fields.get("activeConnections"),
    }
