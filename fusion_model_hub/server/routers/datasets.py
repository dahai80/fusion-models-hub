import json
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ...db import crud
from ..deps import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])

_MAX_LINEAGE_CHAIN = 64


class DatasetCreate(BaseModel):
    # #100: trainer lineage sidecar shape — {source, source_path,
    # source_sha256, transforms: [...], output}. Free-form dict so transform
    # entries can evolve without a hub release; sha256 fields are the only
    # contract the lineage resolver relies on.
    name: str = Field(..., min_length=1, max_length=128)
    sha256: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    num_samples: int = Field(0, ge=0)
    size_bytes: int = Field(0, ge=0)
    lineage: dict = Field(default_factory=dict)


def _dataset_out(d) -> dict:
    try:
        lineage = json.loads(d.lineage) if d.lineage else {}
    except (ValueError, TypeError):
        logger.warning("Dataset %s has malformed lineage JSON; returning raw", d.id)
        lineage = {"raw": d.lineage}
    return {
        "id": d.id,
        "name": d.name,
        "sha256": d.sha256,
        "num_samples": d.num_samples,
        "size_bytes": d.size_bytes,
        "lineage": lineage,
        "tenant_id": d.tenant_id or "",
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "updated_at": d.updated_at.isoformat() if d.updated_at else None,
    }


@router.post("")
async def register_dataset(body: DatasetCreate, session: SessionDep, request: Request):
    # #100: single registration point for trainer dataset records. 409 on an
    # existing (name, sha256) — the caller re-GETs by the id in the detail,
    # mirroring model registration idempotency. Records only; dataset files
    # stay local to the trainer (the record carries the sha256).
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    existing = await crud.get_dataset_by_name_sha(session, body.name, body.sha256)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset already registered: id={existing.id}",
        )
    try:
        lineage_json = json.dumps(body.lineage) if body.lineage else ""
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"lineage is not JSON-serializable: {e}")
    d = await crud.create_dataset(
        session,
        name=body.name,
        sha256=body.sha256.lower(),
        num_samples=body.num_samples,
        size_bytes=body.size_bytes,
        lineage=lineage_json,
        tenant_id=tenant_id,
    )
    logger.info("Dataset registered: id=%s name=%s source=%s", d.id, d.name, body.lineage.get("source", ""))
    return _dataset_out(d)


@router.get("")
async def list_datasets(
    session: SessionDep,
    request: Request,
    keyword: str = "",
    page: int = 1,
    page_size: int = 20,
):
    tenant_id = getattr(request.state, "tenant_id", "") or ""
    datasets, total = await crud.list_datasets(
        session,
        keyword=keyword,
        tenant_id=tenant_id,
        page=page,
        page_size=page_size,
    )
    return {"items": [_dataset_out(d) for d in datasets], "total": total, "page": page, "page_size": page_size}


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, session: SessionDep):
    d = await crud.get_dataset(session, dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return _dataset_out(d)


@router.get("/{dataset_id}/lineage")
async def get_dataset_lineage(dataset_id: str, session: SessionDep):
    # #100: resolve the full ancestor chain server-side. The trainer's sidecar
    # records source_sha256 (the export/sink file the dataset was built from);
    # walking dataset.sha256 == lineage.source_sha256 links registered records
    # into a chain. Unregistered ancestors surface as {"sha256": ...} stubs so
    # the chain shows the gap instead of silently truncating.
    d = await crud.get_dataset(session, dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    chain = [_dataset_out(d)]
    seen_shas = {d.sha256}
    current = d
    while len(chain) < _MAX_LINEAGE_CHAIN:
        try:
            lineage = json.loads(current.lineage) if current.lineage else {}
        except (ValueError, TypeError):
            lineage = {}
        parent_sha = str(lineage.get("source_sha256", "")).lower()
        if not parent_sha or parent_sha in seen_shas:
            break
        seen_shas.add(parent_sha)
        parents = await crud.find_datasets_by_sha(session, parent_sha)
        if not parents:
            chain.append({"sha256": parent_sha, "registered": False})
            break
        parent = parents[0]
        chain.append(_dataset_out(parent))
        current = parent

    if len(chain) >= _MAX_LINEAGE_CHAIN:
        logger.warning("Dataset lineage chain truncated at %d: dataset=%s", _MAX_LINEAGE_CHAIN, dataset_id)
    return {"dataset_id": d.id, "depth": len(chain), "chain": chain}
