import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ...db.models import DeploymentStatus
from ..deps import SessionDep, SettingsDep
from .inference import _mlx_headers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deployments", tags=["deployments"])


async def _load_model_via_mlx(settings, model_name: str) -> bool:
    headers = _mlx_headers(settings)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.mlx_url}/v1/models/{model_name}/load", headers=headers,
            )
            if resp.status_code in (200, 409):
                return True
            if resp.status_code == 404:
                logger.info(
                    "MLX /load 404 for %s (upstream route shadowing), falling back to chat auto-load",
                    model_name,
                )
                chat_resp = await client.post(
                    f"{settings.mlx_url}/v1/chat/completions",
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



class DeploymentCreate(BaseModel):
    model_id: str = Field(..., min_length=1)
    name: str = ""
    version_id: str = ""
    replicas: int = Field(1, ge=1, le=100)
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
    mlx_loaded = await _load_model_via_mlx(settings, model_name)
    dep_name = body.effective_name(model_name)
    d = await crud.create_deployment(
        session, model_id=body.model_id, name=dep_name,
        tenant_id=tenant_id, version_id=body.version_id, replicas=body.effective_replicas,
    )
    initial_status = DeploymentStatus.RUNNING if mlx_loaded else DeploymentStatus.PENDING
    await crud.update_deployment(session, d.id, status=initial_status)
    d = await crud.get_deployment(session, d.id)
    logger.info("Deployment created: id=%s model=%s replicas=%d", d.id, model_name, d.replicas)
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
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{settings.mlx_url}/v1/models/{model_name}/unload",
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
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{settings.mlx_url}/v1/models/{model_name}/unload",
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
    mlx_metrics = {}
    if model_name and d.status == DeploymentStatus.RUNNING:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{settings.mlx_url}/v1/models/status",
                    headers=_mlx_headers(settings),
                )
                if resp.status_code == 200:
                    status_data = resp.json()
                    if isinstance(status_data, dict):
                        mlx_metrics = status_data.get(model_name, status_data)
        except Exception as e:
            logger.warning("Failed to fetch MLX metrics: %s", e)
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
    # Real sources only for avgLatencyMs (version.inference_latency, ms) and
    # tokensPerSecond (version.throughput); others null (studio Double?/Int?
    # decode null -> nil). Emitted always so the shape is stable + testable.
    avg_latency = float(version_metrics.get("inference_latency") or 0.0)
    tps = float(version_metrics.get("throughput") or 0.0)
    return {
        "deployment_id": deployment_id,
        "status": d.status.value,
        "replicas": d.replicas,
        "gray_enabled": d.gray_enabled,
        "mlx_metrics": mlx_metrics,
        "version_metrics": version_metrics,
        "deploymentId": deployment_id,
        "requestsPerSecond": None,
        "avgLatencyMs": round(avg_latency, 2) if avg_latency > 0 else None,
        "errorRate": None,
        "tokensPerSecond": round(tps, 2) if tps > 0 else None,
        "activeConnections": None,
    }
