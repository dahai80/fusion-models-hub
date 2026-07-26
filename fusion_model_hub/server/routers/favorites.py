import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["favorites"])


class FavoriteCreate(BaseModel):
    pass


def _favorite_to_dict(f) -> dict:
    return {
        "id": f.id,
        "model_id": f.model_id,
        "user_id": f.user_id,
        "created_at": f.created_at.isoformat() if f.created_at else None,
    }


@router.post("/{model_id}/favorites", status_code=201)
async def add_favorite(
    model_id: str, session: SessionDep, request: Request,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    user_id = getattr(request.state, "tenant_id", "") or ""
    existing = await crud.is_model_favorited(session, model_id, user_id)
    if existing:
        raise HTTPException(status_code=409, detail="Already favorited")
    f = await crud.create_model_favorite(session, model_id=model_id, user_id=user_id)
    return _favorite_to_dict(f)


@router.get("/{model_id}/favorites")
async def list_favorites(
    model_id: str, session: SessionDep,
    page: int = 1, page_size: int = 20,
):
    m = await crud.get_model(session, model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    favs, total = await crud.list_model_favorites(
        session, model_id=model_id, page=page, page_size=page_size,
    )
    return {
        "items": [_favorite_to_dict(f) for f in favs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/favorites/me")
async def list_my_favorites(
    session: SessionDep, request: Request,
    page: int = 1, page_size: int = 20,
):
    user_id = getattr(request.state, "tenant_id", "") or ""
    favs, total = await crud.list_model_favorites(
        session, user_id=user_id, page=page, page_size=page_size,
    )
    return {
        "items": [_favorite_to_dict(f) for f in favs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.delete("/favorites/{favorite_id}")
async def remove_favorite(favorite_id: str, session: SessionDep):
    ok = await crud.delete_model_favorite(session, favorite_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Favorite not found")
    return {"detail": "deleted"}
