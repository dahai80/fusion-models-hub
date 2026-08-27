import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ...db.models import EvaluationStatus
from ..deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


class EvaluationCreate(BaseModel):
    model_id: str = Field(..., min_length=1)
    version_id: str = ""
    benchmark_name: str = Field(..., min_length=1, max_length=64)


class EvaluationUpdate(BaseModel):
    status: str | None = None
    score: float | None = None
    metrics: str | None = None
    error_message: str | None = None


def _eval_to_dict(e) -> dict:
    return {
        "id": e.id,
        "tenant_id": e.tenant_id,
        "model_id": e.model_id,
        "version_id": e.version_id,
        "benchmark_name": e.benchmark_name,
        "status": e.status.value,
        "score": e.score,
        "metrics": e.metrics,
        "error_message": e.error_message,
        "created_at": e.created_at.isoformat() if e.created_at else None,
        "completed_at": e.completed_at.isoformat() if e.completed_at else None,
    }


@router.post("", status_code=201)
async def create_evaluation(body: EvaluationCreate, session: SessionDep, request: Request):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    e = await crud.create_evaluation(
        session,
        model_id=body.model_id,
        benchmark_name=body.benchmark_name,
        tenant_id=tenant_id,
        version_id=body.version_id,
    )
    # #3: spawn the async runner so the row leaves PENDING. The runner submits
    # a Fusion-Bench task, polls it to completion, and writes score/metrics
    # back. Fire-and-forget — the client polls GET /evaluations/{id}. If the
    # runner cannot start (e.g. bench unreachable) it flips the row to FAILED
    # with a clear error_message; the 201 response still returns the row so
    # the caller has the eval_id to poll.
    try:
        from ..eval_tasks import submit_evaluation

        await submit_evaluation(e.id, body.model_id, body.version_id, body.benchmark_name)
    except Exception:
        logger.exception("Failed to submit evaluation runner: id=%s", e.id)
    return _eval_to_dict(e)


@router.get("")
async def list_evaluations(
    session: SessionDep,
    model_id: str = "",
    version_id: str = "",
    benchmark_name: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 20,
):
    evals, total = await crud.list_evaluations(
        session,
        model_id=model_id,
        version_id=version_id,
        benchmark_name=benchmark_name,
        status=status,
        page=page,
        page_size=page_size,
    )
    return {
        "items": [_eval_to_dict(e) for e in evals],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/benchmarks/compare")
async def compare_benchmarks(
    session: SessionDep,
    model_id: str = "",
    benchmark_name: str = "",
):
    if not model_id or not benchmark_name:
        raise HTTPException(status_code=400, detail="model_id and benchmark_name are required")
    evals, _ = await crud.list_evaluations(
        session,
        model_id=model_id,
        benchmark_name=benchmark_name,
        status=EvaluationStatus.COMPLETED.value,
        page_size=100,
    )
    if not evals:
        raise HTTPException(status_code=404, detail="No completed evaluations found")
    return {
        "model_id": model_id,
        "benchmark_name": benchmark_name,
        "results": [_eval_to_dict(e) for e in evals],
        "best_score": max(e.score for e in evals),
        "average_score": round(sum(e.score for e in evals) / len(evals), 2),
    }


@router.get("/{eval_id}")
async def get_evaluation(eval_id: str, session: SessionDep):
    e = await crud.get_evaluation(session, eval_id)
    if not e:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _eval_to_dict(e)


@router.patch("/{eval_id}")
async def update_evaluation(eval_id: str, body: EvaluationUpdate, session: SessionDep):
    fields = {}
    if body.status is not None:
        try:
            EvaluationStatus(body.status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")
        fields["status"] = EvaluationStatus(body.status)
    if body.score is not None:
        fields["score"] = body.score
    if body.metrics is not None:
        fields["metrics"] = body.metrics
    if body.error_message is not None:
        fields["error_message"] = body.error_message
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    e = await crud.update_evaluation(session, eval_id, **fields)
    if not e:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _eval_to_dict(e)


@router.delete("/{eval_id}")
async def delete_evaluation(eval_id: str, session: SessionDep):
    ok = await crud.delete_evaluation(session, eval_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"detail": "deleted"}
