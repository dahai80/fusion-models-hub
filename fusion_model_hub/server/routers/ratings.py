import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["ratings"])


class RatingCreate(BaseModel):
    score: int = Field(..., ge=1, le=5)
    comment: str = ""


class RatingUpdate(BaseModel):
    score: int | None = Field(None, ge=1, le=5)
    comment: str | None = None


def _rating_to_dict(r) -> dict:
    return {
        "id": r.id,
        "model_id": r.model_id,
        "user_id": r.user_id,
        "score": r.score,
        "comment": r.comment,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@router.post("/{model_id}/ratings", status_code=201)
async def create_rating(
    model_id: str,
    body: RatingCreate,
    session: SessionDep,
    request: Request,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    user_id = getattr(request.state, "tenant_id", "") or ""
    r = await crud.create_model_rating(
        session,
        model_id=model_id,
        user_id=user_id,
        score=body.score,
        comment=body.comment,
    )
    return _rating_to_dict(r)


@router.get("/{model_id}/ratings")
async def list_ratings(
    model_id: str,
    session: SessionDep,
    user_id: str = "",
    page: int = 1,
    page_size: int = 20,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    ratings, total = await crud.list_model_ratings(
        session,
        model_id=model_id,
        user_id=user_id,
        page=page,
        page_size=page_size,
    )
    avg = await crud.get_model_avg_rating(session, model_id)
    return {
        "items": [_rating_to_dict(r) for r in ratings],
        "total": total,
        "page": page,
        "page_size": page_size,
        "average_score": round(avg, 2),
    }


@router.get("/{model_id}/ratings/summary")
async def get_rating_summary(model_id: str, session: SessionDep):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    avg = await crud.get_model_avg_rating(session, model_id)
    _, total = await crud.list_model_ratings(session, model_id=model_id, page_size=1)
    return {
        "model_id": model_id,
        "average_score": round(avg, 2),
        "total_ratings": total,
    }


@router.delete("/ratings/{rating_id}")
async def delete_rating(rating_id: str, session: SessionDep):
    ok = await crud.delete_model_rating(session, rating_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Rating not found")
    return {"detail": "deleted"}
