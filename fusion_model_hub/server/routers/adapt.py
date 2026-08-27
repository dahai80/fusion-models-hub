from __future__ import annotations

import asyncio
import logging
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...adapt.decision import AdaptDecisionEngine
from ...adapt.types import AdaptationLevel, AdaptationResult, MigrationPlan
from ..deps import SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/adapt", tags=["adapt"])

_engine: AdaptDecisionEngine | None = None

_running_executions: dict[str, asyncio.Task] = {}
# H7: track per-execution errors so a convert/quantize non-200 (previously
# logged as a warning then reported as "completed") surfaces honestly instead
# of a silent false success.
_execution_errors: dict[str, str] = {}


def _get_engine(settings: SettingsDep) -> AdaptDecisionEngine:
    global _engine
    # E-E10: invalidate on mlx_url OR mlx_internal_api_key drift, not just mlx_url.
    # A key rotation or a hot-reload that also re-keys MLX must rebuild so the
    # engine authenticates against the new credential. On rebuild, drop the old
    # engine's embedded HardwareDetector 5-min cache so a swapped MLX URL does
    # not keep serving the prior MLX's stale hardware profile.
    if _engine is None or _engine.mlx_url != settings.mlx_url or _engine.api_key != settings.mlx_internal_api_key:
        if _engine is not None:
            _engine.invalidate_cache()
        _engine = AdaptDecisionEngine(settings.mlx_url, api_key=settings.mlx_internal_api_key)
        logger.info(
            "AdaptDecisionEngine (re)built for mlx_url=%s",
            settings.mlx_url,
        )
    return _engine


class AssessRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier or alias")
    hf_repo: str | None = Field(None, description="HuggingFace repo (org/name)")
    source_format: str | None = Field(None, description="Source format: pytorch|safetensors|gguf")


class PlanRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier or alias")
    params_b: float = Field(..., gt=0, description="Model size in billions of parameters")
    hf_repo: str | None = Field(None, description="HuggingFace repo (org/name)")
    source_format: str | None = Field(None, description="Source format: pytorch|safetensors|gguf")


class ExecuteRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier or alias")
    hf_repo: str | None = Field(None, description="HuggingFace repo (org/name)")
    source_format: str | None = Field(None, description="Source format: pytorch|safetensors|gguf")
    quant_bits: int = Field(4, ge=2, le=8, description="Quantization bits (2-8)")
    params_b: float = Field(0, ge=0, description="Model size in billions of parameters")


@router.post("/assess", response_model=AdaptationResult)
async def assess_model(request: AssessRequest, settings: SettingsDep):
    engine = _get_engine(settings)
    try:
        return await engine.assess(request.model_id, request.hf_repo, request.source_format)
    except Exception as e:
        logger.error("Adapt assessment failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Assessment unavailable: {e}")


@router.post("/plan", response_model=MigrationPlan)
async def generate_migration_plan(request: PlanRequest, settings: SettingsDep):
    engine = _get_engine(settings)
    try:
        return await engine.assess_and_plan(
            request.model_id,
            request.params_b,
            request.hf_repo,
            request.source_format,
        )
    except Exception as e:
        logger.error("Migration plan generation failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Plan generation unavailable: {e}")


@router.post("/execute", status_code=202)
async def execute_adaptation(request: ExecuteRequest, settings: SettingsDep):
    execution_id = uuid.uuid4().hex[:16]

    async def _run_pipeline():
        try:
            logger.info("Adapt execution started: id=%s model=%s", execution_id, request.model_id)

            adapt_result = await _get_engine(settings).assess(
                request.model_id,
                request.hf_repo,
                request.source_format,
            )
            if adapt_result.level == AdaptationLevel.L4:
                logger.warning("Adapt execution aborted: model %s is L4 (unsupported)", request.model_id)
                return

            mlx_url = settings.mlx_url.rstrip("/")
            hf_source = request.hf_repo or request.model_id

            async with httpx.AsyncClient(timeout=300.0) as client:
                convert_resp = await client.post(
                    f"{mlx_url}/v1/convert",
                    json={"model": hf_source},
                )
                if convert_resp.status_code not in (200, 202):
                    logger.warning(
                        "Convert step returned %d for %s: %s",
                        convert_resp.status_code,
                        request.model_id,
                        convert_resp.text,
                    )
                    _execution_errors[execution_id] = f"convert failed: MLX returned {convert_resp.status_code}"
                else:
                    logger.info("Convert submitted for %s", request.model_id)

                if request.quant_bits not in (16,):
                    quant_resp = await client.post(
                        f"{mlx_url}/v1/quantize",
                        json={"model": hf_source, "bits": request.quant_bits},
                    )
                    if quant_resp.status_code not in (200, 202):
                        logger.warning(
                            "Quantize step returned %d for %s: %s",
                            quant_resp.status_code,
                            request.model_id,
                            quant_resp.text,
                        )
                        _execution_errors[execution_id] = f"quantize failed: MLX returned {quant_resp.status_code}"
                    else:
                        logger.info("Quantize submitted for %s at %d-bit", request.model_id, request.quant_bits)
                else:
                    # E-E2: quant_bits==16 means no quantization requested — this
                    # /adapt/execute pipeline is a debug passthrough (H7: no
                    # version/cache row, hub_registered=False), so a 16-bit run
                    # legitimately has nothing to quantize. Log it explicitly so
                    # the silent skip is visible in the audit trail rather than
                    # looking like a dropped step.
                    logger.info(
                        "Skipping quantize for %s: quant_bits=16 (no quantization requested, "
                        "debug passthrough converts only)",
                        request.model_id,
                    )

            logger.info("Adapt execution completed: id=%s model=%s", execution_id, request.model_id)
        except Exception:
            logger.exception("Adapt execution failed: id=%s model=%s", execution_id, request.model_id)

    t = asyncio.create_task(_run_pipeline())
    _running_executions[execution_id] = t
    # H7: keep the task in _running_executions after completion so
    # get_execution_status can still report the terminal state (the prior
    # done_callback popped it, making a finished execution 404 and hiding
    # whether it succeeded or failed).

    # H7: this pipeline proxies convert/quantize to MLX and creates no
    # ModelVersion or cache row in the hub — the output (if any) lives in
    # MLX. Surface that the hub does not register the result.
    return {
        "execution_id": execution_id,
        "status": "running",
        "model_id": request.model_id,
        "hub_registered": False,
    }


@router.get("/execute/{execution_id}")
async def get_execution_status(execution_id: str):
    task = _running_executions.get(execution_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    if task.done():
        if task.exception():
            return {"execution_id": execution_id, "status": "failed", "error": str(task.exception())}
        # H7: a clean task exit is NOT success if a convert/quantize step
        # recorded a non-200 — report failed with the recorded reason rather
        # than a silent "completed".
        err = _execution_errors.get(execution_id, "")
        if err:
            return {"execution_id": execution_id, "status": "failed", "error": err}
        return {"execution_id": execution_id, "status": "completed"}
    return {"execution_id": execution_id, "status": "running"}
