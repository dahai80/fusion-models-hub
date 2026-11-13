import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ...catalog.schema import validate_entry
from ..deps import SettingsDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/catalog", tags=["catalog"])


def _load_entries(settings):
    # #102: serve the static catalog over HTTP for consumers that prefer a
    # live read. Files failing the schema are skipped with a loud warning —
    # one bad file must not break the whole listing (CI still fails on it).
    root = Path(settings.catalog_dir) if settings.catalog_dir else None
    if not root or not root.is_dir():
        logger.warning("Catalog dir not available: %s", root)
        return []
    entries = []
    for f in sorted(root.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            entry = validate_entry(data, source=str(f))
        except (ValueError, OSError) as e:
            logger.warning("Skipping invalid catalog file %s: %s", f, e)
            continue
        entries.append(entry.model_dump())
    return entries


@router.get("")
async def list_catalog(settings: SettingsDep):
    entries = _load_entries(settings)
    return {"models": entries, "total": len(entries), "schemaVersion": 1}


@router.get("/{name}")
async def get_catalog_entry(name: str, settings: SettingsDep):
    entries = _load_entries(settings)
    for e in entries:
        if e["name"] == name:
            return e
    raise HTTPException(status_code=404, detail=f"Catalog entry not found: {name}")
