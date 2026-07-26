import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..tasks import get_task_status, list_running_tasks, submit_quantize

logger = logging.getLogger(__name__)
router = APIRouter(tags=["quantize"])


class QuantizeRequest(BaseModel):
    source_version_id: str
    target_format: str = "mlx"
    quant_bits: int = 4


@router.post("/quantize", status_code=202)
async def start_quantize(body: QuantizeRequest):
    if body.quant_bits not in (2, 4, 6, 8):
        raise HTTPException(status_code=400, detail="quant_bits must be one of: 2, 4, 6, 8")
    try:
        task_id = await submit_quantize(
            source_version_id=body.source_version_id,
            target_format=body.target_format,
            quant_bits=body.quant_bits,
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
    from ...db import crud
    from ..deps import get_session_factory
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
