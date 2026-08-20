import asyncio
import logging
import subprocess
import time
from collections import defaultdict

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ...db import crud
from ..deps import SessionDep, SettingsDep, StoreDep

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


def _format_uptime(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m}m"
    d, h = divmod(h, 24)
    return f"{d}d{h}h{m}m"


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
    mlx_headers = {"X-Fusion-Source": "model-hub"}
    if settings.mlx_internal_api_key:
        mlx_headers["Authorization"] = f"Bearer {settings.mlx_internal_api_key}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.mlx_url}/health", headers=mlx_headers)
            if resp.status_code == 200:
                mlx_status = "available"
                try:
                    mlx_info = resp.json()
                except Exception:
                    mlx_info = {"raw": resp.text[:200]}
            else:
                mlx_status = f"error_{resp.status_code}"
    except httpx.ConnectError:
        mlx_status = "offline"
    except Exception as e:
        mlx_status = "error"
        logger.warning("MLX health check failed: %s", e)

    storage = store.get_storage_stats()
    overall = "healthy" if mlx_status == "available" else "degraded"

    # fusion-studio HubHealthResponse expects {status, version, uptime, mlxConnected:Bool, storage:HubDiskStats}
    try:
        from fusion_model_hub import __version__ as hub_version
    except Exception:
        hub_version = "unknown"

    from ..deps import get_start_ts
    start_ts = get_start_ts()
    uptime_str = ""
    if start_ts:
        uptime_str = _format_uptime(time.time() - start_ts)

    used_gb = round(storage.get("total_size_gb", 0.0), 2)
    disk_stats = {
        # studio HubDiskStats shape (all optional)
        "used": used_gb,
        "total": round(used_gb + storage.get("free_gb", 0.0), 2),
        "modelsPath": storage.get("path", ""),
        "modelsSize": used_gb,
        # hub-internal fields kept for existing consumers
        "path": storage.get("path", ""),
        "model_count": storage.get("model_count", 0),
        "file_count": storage.get("file_count", 0),
        "total_size_gb": used_gb,
    }

    return {
        "status": overall,
        "version": hub_version,
        "uptime": uptime_str,
        "mlxConnected": mlx_status == "available",
        "model_count": model_count,
        "mlx": {
            "status": mlx_status,
            "url": settings.mlx_url,
            "info": mlx_info,
        },
        "storage": disk_stats,
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


@router.post("/system/scan-duplicates")
async def scan_duplicate_weights(session: SessionDep):
    all_models, _ = await crud.list_models(session, page_size=10000)

    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for m in all_models:
        for v in m.versions:
            if v.file_hash:
                hash_groups[v.file_hash].append({
                    "model_id": m.id,
                    "model_name": m.name,
                    "version_id": v.id,
                    "version": v.version,
                    "file_hash": v.file_hash,
                    "file_size": v.file_size,
                })

    duplicates = [group for group in hash_groups.values() if len(group) > 1]
    logger.info("Duplicate scan found %d duplicate groups", len(duplicates))
    return {"duplicate_groups": duplicates, "total_groups": len(duplicates)}


@router.post("/system/cleanup")
async def disk_cleanup(session: SessionDep):
    all_models, _ = await crud.list_models(session, page_size=10000)
    candidates = []
    for m in all_models:
        for v in m.versions:
            if v.status.value == "retired" and v.file_path:
                candidates.append({
                    "model_id": m.id,
                    "model_name": m.name,
                    "version_id": v.id,
                    "version": v.version,
                    "file_path": v.file_path,
                    "file_size": v.file_size,
                    "status": v.status.value,
                })
    logger.info("Cleanup scan found %d retired versions with files", len(candidates))
    return {"candidates": candidates, "total": len(candidates)}


def _collect_hardware_info() -> dict:
    gpu_name = ""
    gpu_memory_total_mb = 0
    gpu_memory_used_mb = 0
    gpu_utilization = 0.0
    cpu_cores = 0
    memory_total_gb = 0.0
    memory_used_gb = 0.0

    try:
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if "Chipset Model:" in stripped or "Chipset:" in stripped:
                    gpu_name = stripped.split(":")[-1].strip()
                elif "VRAM (Dynamic, Max):" in stripped or "VRAM (Total):" in stripped:
                    val = stripped.split(":")[-1].strip()
                    if "MB" in val:
                        gpu_memory_total_mb = int("".join(c for c in val if c.isdigit()))
                    elif "GB" in val:
                        gpu_memory_total_mb = int(float("".join(c for c in val if c.isdigit() or c == ".")) * 1024)
    except Exception:
        logger.debug("GPU info collection failed", exc_info=True)

    import os
    cpu_cores = os.cpu_count() or 0

    try:
        result = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            page_size = 16384
            free_pages = 0
            total_pages = 0
            for line in result.stdout.splitlines():
                if "Pages free:" in line:
                    free_pages += int(line.split(":")[-1].strip().rstrip("."))
                elif "Pages active:" in line or "Pages inactive:" in line:
                    total_pages += int(line.split(":")[-1].strip().rstrip("."))
                elif "Pages speculative:" in line:
                    free_pages += int(line.split(":")[-1].strip().rstrip("."))
                elif "Pages wired down:" in line:
                    total_pages += int(line.split(":")[-1].strip().rstrip("."))
            if total_pages + free_pages > 0:
                memory_total_gb = round((total_pages + free_pages) * page_size / (1024 ** 3), 2)
                memory_used_gb = round(total_pages * page_size / (1024 ** 3), 2)
    except Exception:
        logger.debug("Memory info collection failed", exc_info=True)

    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            memory_total_gb = round(int(result.stdout.strip()) / (1024 ** 3), 2)
    except Exception:
        logger.debug("sysctl memsize failed", exc_info=True)

    if not gpu_name:
        try:
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip()
        except Exception:
            logger.debug("CPU brand detection failed", exc_info=True)

    return {
        "gpu_name": gpu_name,
        "gpu_memory_total_mb": gpu_memory_total_mb,
        "gpu_memory_used_mb": gpu_memory_used_mb,
        "gpu_utilization": gpu_utilization,
        "cpu_cores": cpu_cores,
        "memory_total_gb": memory_total_gb,
        "memory_used_gb": memory_used_gb,
    }


@router.get("/system/hardware")
async def hardware_info():
    return await asyncio.to_thread(_collect_hardware_info)
