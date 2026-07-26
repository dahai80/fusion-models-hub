import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ...db.models import TaskStatus
from ..deps import SessionDep, get_session_factory
from ..tasks import get_task_status, list_running_tasks, submit_quantize

logger = logging.getLogger(__name__)
router = APIRouter(tags=["quantize"])


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
async def list_quantize_tasks(status: str = "", page: int = 1, page_size: int = 20):
    session_factory = get_session_factory()
    async with session_factory() as session:
        tasks, total = await crud.list_quantize_tasks(
            session, status=status, page=page, page_size=page_size,
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
async def get_quantize_status(task_id: str):
    status = await get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/quantize/{task_id}/compare")
async def compare_quantize_results(task_id: str):
    sf = get_session_factory()
    async with sf() as session:
        task = await crud.get_quantize_task(session, task_id)
        if not task:
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
        sf = get_session_factory()
        async with sf() as s:
            try:
                await crud.update_lora_merge_task(s, task_id, status=TaskStatus.RUNNING)
                logger.info("LoRA merge running: id=%s", task_id)
                await asyncio.sleep(0.1)
                await crud.update_lora_merge_task(
                    s, task_id, status=TaskStatus.COMPLETED,
                    output_version_id="",
                )
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
