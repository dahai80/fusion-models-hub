import logging

from fastapi import APIRouter, HTTPException, Query

from ...cache.types import CacheLevel
from ..deps import CacheDep, SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


def _stats_to_dict(stats) -> dict:
    return {
        "total_entries": stats.total_entries,
        "total_size_bytes": stats.total_size_bytes,
        "total_size_gb": stats.total_size_gb,
        "raw_count": stats.raw_count,
        "converted_count": stats.converted_count,
        "quantized_count": stats.quantized_count,
        "levels": stats.levels,
    }


@router.get("")
async def cache_stats(cache: CacheDep):
    logger.info("Cache stats requested")
    return _stats_to_dict(cache.stats())


@router.get("/entries")
async def cache_list_entries(
    cache: CacheDep,
    level: str | None = Query(None, description="Filter by level: raw/converted/quantized"),
):
    logger.info("Cache entries list requested: level=%s", level)
    entries = []
    for key, data in cache._index.items():
        if level and data.get("level") != level:
            continue
        entries.append({
            "key": key,
            "model_id": data.get("model_id"),
            "level": data.get("level"),
            "path": data.get("path"),
            "size_bytes": data.get("size_bytes", 0),
            "quant_bits": data.get("quant_bits", 0),
            "mlx_version": data.get("mlx_version", ""),
            "created_at": data.get("created_at", 0),
            "last_accessed": data.get("last_accessed", 0),
        })
    return {"entries": entries, "count": len(entries)}


@router.post("/gc")
async def cache_gc(
    cache: CacheDep,
    max_size_gb: float = Query(0, description="Max size in GB (0 = no size limit)"),
    max_age_days: float = Query(30, description="Max age in days"),
):
    removed = cache.gc(max_size_gb=max_size_gb, max_age_days=max_age_days)
    logger.info("Cache GC removed %d entries", removed)
    return {"removed": removed, "stats": _stats_to_dict(cache.stats())}


@router.post("/validate")
async def cache_validate(
    cache: CacheDep,
    settings: SettingsDep,
    mlx_version: str = Query("", description="Current MLX version for staleness check"),
):
    result = cache.validate(current_mlx_version=mlx_version)
    logger.info("Cache validate: %s", result)
    return result


@router.delete("/{model_id}")
async def cache_remove_model(cache: CacheDep, model_id: str):
    removed = cache.remove_model(model_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No cache entries for model {model_id}")
    logger.info("Removed %d cache entries for model %s", removed, model_id)
    return {"removed": removed, "model_id": model_id}


@router.delete("/{model_id}/{level}")
async def cache_remove_entry(
    cache: CacheDep,
    model_id: str,
    level: str,
    quant_bits: int = Query(0, description="Quant bits (only for quantized level)"),
):
    try:
        cache_level = CacheLevel(level)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid level: {level}")
    ok = cache.remove(model_id, cache_level, quant_bits=quant_bits)
    if not ok:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    logger.info("Removed cache entry: model=%s level=%s bits=%d", model_id, level, quant_bits)
    return {"removed": True, "model_id": model_id, "level": level, "quant_bits": quant_bits}
