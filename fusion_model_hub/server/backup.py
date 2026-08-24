import asyncio
import json
import logging
import os
from datetime import UTC, datetime

from ..db.crud import list_models, list_versions
from .deps import get_session_factory, get_settings

logger = logging.getLogger(__name__)

_backup_task: asyncio.Task | None = None


async def _run_backup_loop() -> None:
    settings = get_settings()
    if not settings.backup_dir:
        logger.info("Backup dir not configured, auto-backup disabled")
        return
    os.makedirs(settings.backup_dir, exist_ok=True)
    logger.info("Auto-backup enabled: dir=%s interval=%ds", settings.backup_dir, settings.backup_interval_seconds)
    while True:
        try:
            await asyncio.sleep(settings.backup_interval_seconds)
            await _perform_backup(settings.backup_dir)
        except asyncio.CancelledError:
            logger.info("Backup loop cancelled")
            return
        except Exception:
            logger.exception("Auto-backup failed")


BACKUP_MAX_FILES = 20
_BACKUP_PAGE_SIZE = 200


async def _perform_backup(backup_dir: str) -> None:
    sf = get_session_factory()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"backup_{timestamp}.json")
    data: dict = {"timestamp": timestamp, "models": [], "versions": []}
    async with sf() as session:
        page = 1
        while True:
            models, total = await list_models(session, page=page, page_size=_BACKUP_PAGE_SIZE)
            if not models:
                break
            for m in models:
                data["models"].append({
                    "id": m.id, "name": m.name, "description": m.description,
                    "model_type": m.model_type.value, "architecture": m.architecture,
                    "params_size": m.params_size, "license": m.license,
                })
                vpage = 1
                while True:
                    versions, vtotal = await list_versions(
                        session, m.id, page=vpage, page_size=_BACKUP_PAGE_SIZE,
                    )
                    if not versions:
                        break
                    for v in versions:
                        data["versions"].append({
                            "id": v.id, "model_id": v.model_id, "version": v.version,
                            "format": v.format.value, "quantization": v.quantization.value,
                            "status": v.status.value, "file_hash": v.file_hash,
                            "file_size": v.file_size, "benchmark_score": v.benchmark_score,
                        })
                    if vpage * _BACKUP_PAGE_SIZE >= vtotal:
                        break
                    vpage += 1
            if page * _BACKUP_PAGE_SIZE >= total:
                break
            page += 1
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _rotate_backups(backup_dir)
    logger.info(
        "Auto-backup completed: file=%s models=%d versions=%d",
        backup_file, len(data["models"]), len(data["versions"]),
    )


def _rotate_backups(backup_dir: str) -> None:
    try:
        files = [
            os.path.join(backup_dir, fn)
            for fn in os.listdir(backup_dir)
            if fn.startswith("backup_") and fn.endswith(".json")
        ]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for stale in files[BACKUP_MAX_FILES:]:
            os.remove(stale)
            logger.info("Rotated out old backup: %s", stale)
    except Exception:
        logger.warning("Backup rotation failed", exc_info=True)


def start_backup_scheduler() -> None:
    global _backup_task
    _backup_task = asyncio.create_task(_run_backup_loop(), name="auto-backup")


def stop_backup_scheduler() -> None:
    global _backup_task
    if _backup_task and not _backup_task.done():
        _backup_task.cancel()
        _backup_task = None
        logger.info("Backup scheduler stopped")
