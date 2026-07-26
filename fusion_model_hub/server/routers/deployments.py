import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ...db.models import DeploymentStatus
from ..deps import SessionDep, SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deployments", tags=["deployments"])


class DeploymentCreate(BaseModel):
    model_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=64)
    version_id: str = ""
    replicas: int = Field(1, ge=1, le=100)


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


@router.post("", status_code=201, response_model=DeploymentOut)
async def create_deployment(body: DeploymentCreate, session: SessionDep, request: Request, settings: SettingsDep):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    model_name = m.hf_repo or m.name
    mlx_loaded = False
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.mlx_url}/v1/models/{model_name}/load")
            if resp.status_code not in (200, 409):
                logger.warning("MLX load returned %d for %s", resp.status_code, model_name)
            else:
                mlx_loaded = True
    except httpx.ConnectError:
        logger.warning("Fusion-MLX server unavailable, deployment created in pending state")
    except Exception as e:
        logger.warning("MLX load call failed: %s", e)
    d = await crud.create_deployment(
        session, model_id=body.model_id, name=body.name,
        tenant_id=tenant_id, version_id=body.version_id, replicas=body.replicas,
    )
    initial_status = DeploymentStatus.RUNNING if mlx_loaded else DeploymentStatus.PENDING
    await crud.update_deployment(session, d.id, status=initial_status)
    d = await crud.get_deployment(session, d.id)
    return d


@router.get("", response_model=list[DeploymentOut])
async def list_deployments(
    session: SessionDep,
    request: Request,
    model_id: str = "",
    status: str = "",
):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    return await crud.list_deployments(session, model_id=model_id, status=status, tenant_id=tenant_id)


@router.get("/{deployment_id}", response_model=DeploymentOut)
async def get_deployment(deployment_id: str, session: SessionDep):
    d = await crud.get_deployment(session, deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return d


@router.patch("/{deployment_id}", response_model=DeploymentOut)
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
    return d


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
                await client.post(f"{settings.mlx_url}/v1/models/{model_name}/unload")
        except Exception as e:
            logger.warning("MLX unload call failed: %s", e)
    await crud.delete_deployment(session, deployment_id)
    return {"detail": "deleted"}


class GrayReleaseRequest(BaseModel):
    gray_version_id: str = Field(..., min_length=1)
    gray_traffic_ratio: int = Field(10, ge=1, le=100)


@router.post("/{deployment_id}/gray", response_model=DeploymentOut)
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
    return d


@router.delete("/{deployment_id}/gray", response_model=DeploymentOut)
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
    return d


class ScaleRequest(BaseModel):
    replicas: int = Field(..., ge=1, le=100)


@router.post("/{deployment_id}/scale", response_model=DeploymentOut)
async def scale_deployment(deployment_id: str, body: ScaleRequest, session: SessionDep, settings: SettingsDep):
    d = await crud.get_deployment(session, deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    d = await crud.update_deployment(session, deployment_id, replicas=body.replicas)
    if body.replicas > 0 and d.status == DeploymentStatus.RUNNING:
        m = await crud.get_model(session, d.model_id)
        if m:
            model_name = m.hf_repo or m.name
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    await client.post(f"{settings.mlx_url}/v1/models/{model_name}/load")
            except Exception as e:
                logger.warning("MLX load on scale failed: %s", e)
    logger.info("Deployment scaled: id=%s replicas=%d", deployment_id, body.replicas)
    return d


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
                resp = await client.get(f"{settings.mlx_url}/v1/models/status")
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
    return {
        "deployment_id": deployment_id,
        "status": d.status.value,
        "replicas": d.replicas,
        "gray_enabled": d.gray_enabled,
        "mlx_metrics": mlx_metrics,
        "version_metrics": version_metrics,
    }
