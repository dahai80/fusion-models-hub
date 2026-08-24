import asyncio
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ...db.models import TaskStatus
from ..deps import SessionDep, SettingsDep, get_session_factory
from ..tasks import get_task_status, list_running_tasks, submit_quantize

logger = logging.getLogger(__name__)
router = APIRouter(tags=["quantize"])


def _caller_tenant(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or ""


class QuantizeRequest(BaseModel):
    source_version_id: str
    target_format: str = "mlx"
    quant_bits: int = 4
    calibration_dataset: str = ""


@router.post("/quantize", status_code=202)
async def start_quantize(body: QuantizeRequest):
    if body.quant_bits not in (2, 4, 6, 8):
        raise HTTPException(status_code=400, detail="quant_bits must be one of: 2, 4, 6, 8")
    try:
        task_id = await submit_quantize(
            source_version_id=body.source_version_id,
            target_format=body.target_format,
            quant_bits=body.quant_bits,
            calibration_dataset=body.calibration_dataset,
        )
    except Exception as e:
        logger.exception("Failed to submit quantize task")
        raise HTTPException(status_code=500, detail=str(e))
    return {"task_id": task_id, "status": "submitted"}


@router.get("/quantize/running")
async def running_quantize_tasks():
    return {"tasks": list_running_tasks()}


@router.get("/quantize")
async def list_quantize_tasks(status: str = "", page: int = 1, page_size: int = 20, request: Request = None):
    session_factory = get_session_factory()
    tenant_id = _caller_tenant(request) if request else ""
    async with session_factory() as session:
        tasks, total = await crud.list_quantize_tasks(
            session, status=status, page=page, page_size=page_size, tenant_id=tenant_id,
        )
        items = [
            {
                "id": t.id,
                "source_version_id": t.source_version_id,
                "target_format": t.target_format,
                "quant_bits": t.quant_bits,
                "calibration_dataset": t.calibration_dataset,
                "status": t.status.value,
                "output_version_id": t.output_version_id,
                "error_message": t.error_message,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ]
        return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/quantize/{task_id}")
async def get_quantize_status(task_id: str, request: Request = None):
    status = await get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    # F-04: cross-tenant guard. Empty caller tenant (local mode) is permissive.
    tenant_id = _caller_tenant(request) if request else ""
    if tenant_id:
        sf = get_session_factory()
        async with sf() as session:
            if await crud.quantize_task_tenant(session, task_id) != tenant_id:
                raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/quantize/{task_id}/compare")
async def compare_quantize_results(task_id: str, request: Request = None):
    sf = get_session_factory()
    async with sf() as session:
        task = await crud.get_quantize_task(session, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        # F-04: cross-tenant guard.
        tenant_id = _caller_tenant(request) if request else ""
        if tenant_id and await crud.quantize_task_tenant(session, task_id) != tenant_id:
            raise HTTPException(status_code=404, detail="Task not found")
        source_ver = await crud.get_version(session, task.source_version_id)
        if not source_ver:
            raise HTTPException(status_code=404, detail="Source version not found")
        result = {
            "task_id": task.id,
            "source_version": {
                "id": source_ver.id,
                "version": source_ver.version,
                "quantization": source_ver.quantization.value,
                "file_size": source_ver.file_size,
                "benchmark_score": source_ver.benchmark_score,
                "inference_latency": source_ver.inference_latency,
                "throughput": source_ver.throughput,
                "memory_usage": source_ver.memory_usage,
            },
        }
        if task.output_version_id:
            output_ver = await crud.get_version(session, task.output_version_id)
            if output_ver:
                result["output_version"] = {
                    "id": output_ver.id,
                    "version": output_ver.version,
                    "quantization": output_ver.quantization.value,
                    "file_size": output_ver.file_size,
                    "benchmark_score": output_ver.benchmark_score,
                    "inference_latency": output_ver.inference_latency,
                    "throughput": output_ver.throughput,
                    "memory_usage": output_ver.memory_usage,
                }
                result["comparison"] = {
                    "size_reduction_pct": round(
                        (1 - output_ver.file_size / source_ver.file_size) * 100, 2
                    ) if source_ver.file_size > 0 else 0,
                    "latency_change_pct": round(
                        (output_ver.inference_latency - source_ver.inference_latency)
                        / source_ver.inference_latency * 100, 2
                    ) if source_ver.inference_latency > 0 else 0,
                    "throughput_change_pct": round(
                        (output_ver.throughput - source_ver.throughput) / source_ver.throughput * 100, 2
                    ) if source_ver.throughput > 0 else 0,
                    "memory_change_pct": round(
                        (output_ver.memory_usage - source_ver.memory_usage) / source_ver.memory_usage * 100, 2
                    ) if source_ver.memory_usage > 0 else 0,
                }
        return result


class LoraMergeRequest(BaseModel):
    base_version_id: str
    lora_version_id: str
    target_format: str = "mlx"
    quant_bits: int = 4


class LayerRule(BaseModel):
    pattern: str
    bits: int


class LayeredQuantizeRequest(BaseModel):
    model: str
    output_path: str | None = None
    default_bits: int = Field(4, ge=2, le=8)
    layer_rules: list[LayerRule] = Field(..., min_length=1)
    quant_group_size: int = Field(64, ge=1)
    quant_mode: str = "affine"
    trust_remote_code: bool = False


class QuantizeEvaluateRequest(BaseModel):
    source_version_id: str
    quant_bits: int = 4
    sample_size: int = 128


_running_lora_merges: dict[str, asyncio.Task] = {}


@router.post("/quantize/lora-merge", status_code=202)
async def start_lora_merge(body: LoraMergeRequest, session: SessionDep):
    if body.quant_bits not in (2, 4, 6, 8):
        raise HTTPException(status_code=400, detail="quant_bits must be one of: 2, 4, 6, 8")
    base_v = await crud.get_version(session, body.base_version_id)
    if not base_v:
        raise HTTPException(status_code=404, detail="Base version not found")
    lora_v = await crud.get_version(session, body.lora_version_id)
    if not lora_v:
        raise HTTPException(status_code=404, detail="LoRA version not found")
    task = await crud.create_lora_merge_task(
        session, base_version_id=body.base_version_id,
        lora_version_id=body.lora_version_id,
        target_format=body.target_format, quant_bits=body.quant_bits,
    )

    async def _run_merge(task_id: str):
        # #22 LoRA adapter merge: call Fusion-MLX POST /v1/merge-adapter (upstream
        # #584), which loads the base + adapter, fuses LoRA/DoRA layers, and saves
        # the merged weights. Then create a new ModelVersion carrying the merged
        # output and record its id as output_version_id.
        from ..deps import get_settings
        from .webhooks import dispatch_webhook_event
        sf = get_session_factory()
        async with sf() as s:
            try:
                await crud.update_lora_merge_task(s, task_id, status=TaskStatus.RUNNING)
                logger.info("LoRA merge running: id=%s", task_id)
                merge_task = await crud.get_lora_merge_task(s, task_id)
                base_v = await crud.get_version(s, merge_task.base_version_id)
                lora_v = await crud.get_version(s, merge_task.lora_version_id)
                base_m = await crud.get_model(s, base_v.model_id)
                model_name = base_m.hf_repo or base_m.name
                settings = get_settings()
                headers = {"X-Fusion-Source": "model-hub"}
                if settings.mlx_internal_api_key:
                    headers["Authorization"] = f"Bearer {settings.mlx_internal_api_key}"
                merge_payload = {
                    "model": model_name,
                    "adapter_path": lora_v.file_path or lora_v.version,
                }
                output_path = ""
                try:
                    async with httpx.AsyncClient(timeout=300.0) as client:
                        resp = await client.post(
                            f"{settings.mlx_url}/v1/merge-adapter",
                            json=merge_payload,
                            headers=headers,
                        )
                        if resp.status_code == 404:
                            raise RuntimeError(
                                "Fusion-MLX has no /v1/merge-adapter endpoint; "
                                "upgrade fusion-mlx to a version with adapter merge support"
                            )
                        resp.raise_for_status()
                        body = resp.json() if resp.content else {}
                        output_path = body.get("output_path", "")
                except httpx.ConnectError as e:
                    raise RuntimeError(f"Fusion-MLX server unavailable: {e}") from e
                new_ver = await crud.create_version(
                    s,
                    model_id=base_v.model_id,
                    version=f"lora-merge-{task_id[:8]}",
                    release_notes=f"LoRA merge of {lora_v.version} onto {base_v.version}",
                    file_path=output_path,
                )
                await crud.update_lora_merge_task(
                    s, task_id, status=TaskStatus.COMPLETED,
                    output_version_id=new_ver.id if new_ver else "",
                )
                logger.info(
                    "LoRA merge completed: id=%s output_version=%s",
                    task_id, new_ver.id if new_ver else "",
                )
                try:
                    await dispatch_webhook_event(
                        "adapter.merged",
                        {
                            "task_id": task_id,
                            "base_version_id": merge_task.base_version_id,
                            "lora_version_id": merge_task.lora_version_id,
                            "output_version_id": new_ver.id if new_ver else "",
                            "model_id": base_v.model_id,
                        },
                        tenant_id=getattr(base_m, "tenant_id", "") or "",
                    )
                except Exception:
                    logger.warning("adapter.merged webhook dispatch failed: id=%s", task_id)
            except Exception as e:
                await crud.update_lora_merge_task(
                    s, task_id, status=TaskStatus.FAILED,
                    error_message=str(e),
                )
                logger.exception("LoRA merge failed: id=%s", task_id)

    t = asyncio.create_task(_run_merge(task.id))
    _running_lora_merges[task.id] = t
    t.add_done_callback(lambda _: _running_lora_merges.pop(task.id, None))
    return {"task_id": task.id, "status": "submitted"}


@router.get("/quantize/lora-merge/{task_id}")
async def get_lora_merge_status(task_id: str, session: SessionDep):
    task = await crud.get_lora_merge_task(session, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="LoRA merge task not found")
    return {
        "id": task.id,
        "base_version_id": task.base_version_id,
        "lora_version_id": task.lora_version_id,
        "target_format": task.target_format,
        "quant_bits": task.quant_bits,
        "status": task.status.value,
        "output_version_id": task.output_version_id,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


@router.post("/quantize/layered", status_code=202)
async def start_layered_quantize(body: LayeredQuantizeRequest, settings: SettingsDep):
    mlx_url = settings.mlx_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "model": body.model,
                "default_bits": body.default_bits,
                "layer_rules": [{"pattern": r.pattern, "bits": r.bits} for r in body.layer_rules],
                "quant_group_size": body.quant_group_size,
                "quant_mode": body.quant_mode,
                "trust_remote_code": body.trust_remote_code,
            }
            if body.output_path:
                payload["output_path"] = body.output_path

            resp = await client.post(
                f"{mlx_url}/v1/quantize/layered",
                json=payload,
            )
            if resp.status_code in (200, 202):
                data = resp.json()
                logger.info("Layered quantize submitted via MLX: model=%s job_id=%s", body.model, data.get("job_id"))
                return {"job_id": data.get("job_id", ""), "status": "submitted"}
            logger.warning("MLX layered quantize returned %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError as e:
        logger.error("MLX not available for layered quantize: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Layered quantize failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quantize/layered/jobs/{job_id}")
async def get_layered_quantize_job(job_id: str, settings: SettingsDep):
    mlx_url = settings.mlx_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{mlx_url}/v1/quantize/layered/jobs/{job_id}")
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Layered quantize job not found")
            logger.warning("MLX layered job status returned %d", resp.status_code)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError as e:
        logger.error("MLX not available: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Layered quantize job status failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quantize/layered/jobs")
async def list_layered_quantize_jobs(settings: SettingsDep):
    mlx_url = settings.mlx_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{mlx_url}/v1/quantize/layered/jobs")
            if resp.status_code == 200:
                return resp.json()
            logger.warning("MLX layered jobs list returned %d", resp.status_code)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError as e:
        logger.error("MLX not available: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Layered quantize jobs list failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantize/evaluate")
async def evaluate_quantize(body: QuantizeEvaluateRequest, settings: SettingsDep):
    mlx_url = settings.mlx_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{mlx_url}/v1/quantize/evaluate",
                json={
                    "source_version_id": body.source_version_id,
                    "quant_bits": body.quant_bits,
                    "sample_size": body.sample_size,
                },
            )
            if resp.status_code == 200:
                logger.info("Quantize evaluation completed for %s", body.source_version_id)
                return resp.json()
            logger.warning("MLX quantize evaluate returned %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except httpx.ConnectError as e:
        logger.error("MLX not available for quantize evaluate: %s", e)
        raise HTTPException(status_code=503, detail="Fusion-MLX not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Quantize evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))


class BatchQuantizeItem(BaseModel):
    source_version_id: str
    quant_bits: int = 4


class BatchQuantizeRequest(BaseModel):
    items: list[BatchQuantizeItem]


@router.post("/quantize/batch", status_code=202)
async def batch_quantize(body: BatchQuantizeRequest):
    task_ids = []
    errors = []
    for item in body.items:
        if item.quant_bits not in (2, 4, 6, 8):
            errors.append({
                "source_version_id": item.source_version_id,
                "error": f"quant_bits must be one of: 2, 4, 6, 8, got {item.quant_bits}",
            })
            continue
        try:
            task_id = await submit_quantize(
                source_version_id=item.source_version_id,
                quant_bits=item.quant_bits,
            )
            task_ids.append({"source_version_id": item.source_version_id, "task_id": task_id})
        except Exception as e:
            logger.exception("Batch quantize item failed: %s", item.source_version_id)
            errors.append({"source_version_id": item.source_version_id, "error": str(e)})
    logger.info("Batch quantize submitted: %d tasks, %d errors", len(task_ids), len(errors))
    return {"task_ids": task_ids, "errors": errors}
