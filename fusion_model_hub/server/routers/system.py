import logging

import httpx
from fastapi import APIRouter

from ...db import crud
from ..deps import SessionDep, StoreDep, SettingsDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/system/health")
async def health_check(session: SessionDep, store: StoreDep, settings: SettingsDep):
    model_count = 0
    try:
        _, total = await crud.list_models(session, page_size=1)
        model_count = total
    except Exception:
        pass

    mlx_status = "unavailable"
    mlx_info: dict = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.mlx_url}/api/health")
            if resp.status_code == 200:
                mlx_status = "available"
                mlx_info = resp.json()
            else:
                mlx_status = f"error_{resp.status_code}"
    except httpx.ConnectError:
        mlx_status = "offline"
    except Exception as e:
        mlx_status = f"error"
        logger.warning("MLX health check failed: %s", e)

    storage = store.get_storage_stats()
    overall = "healthy" if mlx_status == "available" else "degraded"
    return {
        "status": overall,
        "model_count": model_count,
        "mlx": {
            "status": mlx_status,
            "url": settings.mlx_url,
            "info": mlx_info,
        },
        "storage": storage,
        "data_dir": settings.data_dir,
    }


@router.get("/system/storage")
async def storage_stats(store: StoreDep):
    return store.get_storage_stats()


@router.get("/system/audit")
async def audit_logs(session: SessionDep, resource_type: str = "", action: str = "", page: int = 1, page_size: int = 20):
    logs, total = await crud.list_audit_logs(
        session, resource_type=resource_type, action=action, page=page, page_size=page_size,
    )
    items = [
        {
            "id": l.id,
            "action": l.action,
            "resource_type": l.resource_type,
            "resource_id": l.resource_id,
            "api_key_id": l.api_key_id,
            "detail": l.detail,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        }
        for l in logs
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}
