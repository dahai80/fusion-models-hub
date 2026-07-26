import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...db import crud
from ..deps import SessionDep, SettingsDep, StoreDep

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
        mlx_status = "error"
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
async def audit_logs(
    session: SessionDep, request: Request, resource_type: str = "",
    action: str = "", page: int = 1, page_size: int = 20,
):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    logs, total = await crud.list_audit_logs(
        session, tenant_id=tenant_id, resource_type=resource_type, action=action, page=page, page_size=page_size,
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


@router.get("/system/export")
async def export_data(session: SessionDep, models: str = ""):
    model_ids = [x.strip() for x in models.split(",") if x.strip()] if models else []
    all_models, _ = await crud.list_models(session, page=1, page_size=10000)
    if model_ids:
        all_models = [m for m in all_models if m.id in model_ids]
    tenants = await crud.list_tenants(session)
    webhooks = await crud.list_webhooks(session)
    export = {
        "version": "1.0",
        "models": [
            {
                "id": m.id, "name": m.name, "tenant_id": m.tenant_id,
                "description": m.description, "model_type": m.model_type.value,
                "architecture": m.architecture, "params_size": m.params_size,
                "license": m.license, "author": m.author, "language": m.language,
                "task_types": m.task_types, "owner": m.owner, "hf_repo": m.hf_repo,
                "tags": [{"key": t.key, "value": t.value} for t in m.tags],
            }
            for m in all_models
        ],
        "tenants": [
            {"id": t.id, "name": t.name, "display_name": t.display_name}
            for t in tenants
        ],
        "webhooks": [
            {"id": w.id, "name": w.name, "url": w.url, "events": w.events, "tenant_id": w.tenant_id}
            for w in webhooks
        ],
    }
    logger.info("Exported data: %d models, %d tenants, %d webhooks", len(all_models), len(tenants), len(webhooks))
    return JSONResponse(content=export)


@router.post("/system/import")
async def import_data(session: SessionDep, data: dict):
    count = 0
    tenants_data = data.get("tenants", [])
    for t in tenants_data:
        existing = await crud.get_tenant_by_name(session, t.get("name", ""))
        if not existing:
            await crud.create_tenant(session, name=t["name"], display_name=t.get("display_name", ""))
            count += 1

    models_data = data.get("models", [])
    for m in models_data:
        existing = await crud.get_model_by_name(session, m.get("name", ""))
        if not existing:
            from ...db.models import ModelType
            try:
                mt = ModelType(m.get("model_type", "llm"))
            except ValueError:
                mt = ModelType.LLM
            new_m = await crud.create_model(
                session, name=m["name"], tenant_id=m.get("tenant_id", ""),
                description=m.get("description", ""), model_type=mt,
                architecture=m.get("architecture", ""), params_size=m.get("params_size", ""),
                license=m.get("license", ""), author=m.get("author", ""),
                language=m.get("language", ""), task_types=m.get("task_types", ""),
                owner=m.get("owner", ""), hf_repo=m.get("hf_repo", ""),
            )
            tags = m.get("tags", [])
            if tags:
                await crud.set_tags(session, new_m.id, tags)
            count += 1

    webhooks_data = data.get("webhooks", [])
    for w in webhooks_data:
        await crud.create_webhook(
            session, name=w["name"], url=w["url"],
            events=w.get("events", ""), tenant_id=w.get("tenant_id", ""),
        )
        count += 1

    logger.info("Imported data: %d items created", count)
    return {"imported": count}
