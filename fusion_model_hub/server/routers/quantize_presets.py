import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep
from ..tasks import submit_quantize

logger = logging.getLogger(__name__)
router = APIRouter(tags=["quantize-presets"])

QUANTIZE_PRESETS = {
    "chat": {
        "name": "chat",
        "description": "4-bit quantization with group_size 64, optimized for chat models",
        "target_format": "mlx",
        "quant_bits": 4,
        "quant_group_size": 64,
        "calibration_dataset": "",
    },
    "code": {
        "name": "code",
        "description": "4-bit quantization with group_size 32 and calibration, optimized for code models",
        "target_format": "mlx",
        "quant_bits": 4,
        "quant_group_size": 32,
        "calibration_dataset": "code",
    },
    "embedding": {
        "name": "embedding",
        "description": "8-bit quantization, optimized for embedding models",
        "target_format": "mlx",
        "quant_bits": 8,
        "quant_group_size": 64,
        "calibration_dataset": "",
    },
}


class PresetApplyRequest(BaseModel):
    source_version_id: str


@router.get("/quantize/presets")
async def list_presets():
    return {"presets": list(QUANTIZE_PRESETS.values())}


@router.post("/quantize/presets/{name}/apply", status_code=202)
async def apply_preset(name: str, body: PresetApplyRequest, session: SessionDep):
    preset = QUANTIZE_PRESETS.get(name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")

    v = await crud.get_version(session, body.source_version_id)
    if not v:
        raise HTTPException(status_code=404, detail="Source version not found")

    if preset["quant_bits"] not in (2, 4, 6, 8):
        raise HTTPException(status_code=400, detail="quant_bits must be one of: 2, 4, 6, 8")

    # P1-1: previously created a PENDING task via crud.create_quantize_task but
    # never spawned the runner — preset tasks stuck PENDING forever. Route
    # through submit_quantize so the task is created AND the runner launched,
    # matching POST /quantize behavior.
    try:
        task_id = await submit_quantize(
            source_version_id=body.source_version_id,
            target_format=preset["target_format"],
            quant_bits=preset["quant_bits"],
            calibration_dataset=preset["calibration_dataset"],
        )
    except Exception as e:
        logger.exception("Failed to submit preset quantize task: preset=%s", name)
        raise HTTPException(status_code=500, detail="Failed to submit quantize task") from e
    logger.info("Applied preset '%s' -> submitted quantize task: id=%s", name, task_id)
    return {
        "task_id": task_id,
        "preset": name,
        "status": "submitted",
        "quant_bits": preset["quant_bits"],
    }
