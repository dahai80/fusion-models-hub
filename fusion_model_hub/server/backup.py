import asyncio
import json
import logging
import os
from datetime import UTC, datetime

from sqlalchemy import select

from ..db.crud import list_models
from ..db.models import ModelVersion
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


def _write_backup_sync(backup_file: str, data: dict) -> None:
    # P1-14: offload the (potentially large) JSON serialize + write off the
    # event loop.
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def _perform_backup(backup_dir: str) -> None:
    sf = get_session_factory()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"backup_{timestamp}.json")
    data: dict = {"timestamp": timestamp, "models": [], "versions": []}
    async with sf() as session:
        # P1-14: kill the N+1. Before, each model ran its own paginated
        # list_versions query (one count + one select per model, per page) —
        # N models = 2N+ round-trips. Fetch ALL versions in one query and group
        # in memory; models stay paginated only because list_models enforces
        # MAX_PAGE_SIZE and tenant scoping we do not want to re-implement here.
        all_versions: list[ModelVersion] = list(
            (await session.execute(select(ModelVersion).order_by(ModelVersion.created_at.desc()))).scalars().all()
        )
        versions_by_model: dict[str, list[ModelVersion]] = {}
        for v in all_versions:
            versions_by_model.setdefault(v.model_id, []).append(v)

        page = 1
        while True:
            models, total = await list_models(session, page=page, page_size=_BACKUP_PAGE_SIZE)
            if not models:
                break
            for m in models:
                data["models"].append(
                    {
                        "id": m.id,
                        "name": m.name,
                        "description": m.description,
                        "model_type": m.model_type.value,
                        "architecture": m.architecture,
                        "params_size": m.params_size,
                        "license": m.license,
                    }
                )
                for v in versions_by_model.get(m.id, []):
                    data["versions"].append(
                        {
                            "id": v.id,
                            "model_id": v.model_id,
                            "version": v.version,
                            "format": v.format.value,
                            "quantization": v.quantization.value,
                            "status": v.status.value,
                            "file_hash": v.file_hash,
                            "file_size": v.file_size,
                            "benchmark_score": v.benchmark_score,
                        }
                    )
            if page * _BACKUP_PAGE_SIZE >= total:
                break
            page += 1
    import anyio

    await anyio.to_thread.run_sync(_write_backup_sync, backup_file, data)
    _rotate_backups(backup_dir)
    logger.info(
        "Auto-backup completed: file=%s models=%d versions=%d",
        backup_file,
        len(data["models"]),
        len(data["versions"]),
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


async def restore_from_backup(backup_file: str) -> dict:
    # P1-22: the auto-backup wrote models+versions to JSON but nothing could
    # read it back — the only ingestion path was the `import` subcommand, whose
    # schema (tenants/webhooks, no versions) does not match the backup schema.
    # A restore was therefore impossible, making the backups dead weight. This
    # reads the backup schema verbatim and re-inserts models + versions,
    # preserving their IDs (so version.model_id FKs stay valid) and skipping
    # rows that already exist (idempotent re-run). Run via the `restore` CLI
    # subcommand against a fresh DB after a failure.
    import anyio

    from ..db.models import Model, ModelFormat, ModelStatus, ModelType, ModelVersion, Quantization

    def _read_backup_sync(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    data = await anyio.to_thread.run_sync(_read_backup_sync, backup_file)
    models_in = data.get("models", [])
    versions_in = data.get("versions", [])
    if not models_in and not versions_in:
        logger.warning("Backup file has no models/versions: %s", backup_file)
        return {"models_restored": 0, "versions_restored": 0, "skipped": 0}

    sf = get_session_factory()
    restored_models = 0
    restored_versions = 0
    skipped = 0
    async with sf() as session:
        existing_model_ids: set[str] = set()
        if models_in:
            rows = (await session.execute(select(Model))).scalars().all()
            existing_model_ids = {m.id for m in rows}
        existing_version_ids: set[str] = set()
        if versions_in:
            vrows = (await session.execute(select(ModelVersion))).scalars().all()
            existing_version_ids = {v.id for v in vrows}
        for m in models_in:
            mid = m.get("id", "")
            if not mid or mid in existing_model_ids:
                skipped += 1
                continue
            try:
                mt = ModelType(m.get("model_type", "llm"))
            except ValueError:
                mt = ModelType.LLM
            session.add(
                Model(
                    id=mid,
                    name=m.get("name", ""),
                    description=m.get("description", ""),
                    model_type=mt,
                    architecture=m.get("architecture", ""),
                    params_size=m.get("params_size", ""),
                    license=m.get("license", ""),
                )
            )
            existing_model_ids.add(mid)
            restored_models += 1
        for v in versions_in:
            vid = v.get("id", "")
            if not vid or vid in existing_version_ids:
                skipped += 1
                continue
            try:
                fmt = ModelFormat(v.get("format", "mlx"))
            except ValueError:
                fmt = ModelFormat.MLX
            try:
                quant = Quantization(v.get("quantization", "4bit"))
            except ValueError:
                quant = Quantization.Q4
            try:
                status = ModelStatus(v.get("status", "draft"))
            except ValueError:
                status = ModelStatus.DRAFT
            session.add(
                ModelVersion(
                    id=vid,
                    model_id=v.get("model_id", ""),
                    version=v.get("version", ""),
                    format=fmt,
                    quantization=quant,
                    status=status,
                    file_hash=v.get("file_hash", ""),
                    file_size=v.get("file_size", 0),
                    benchmark_score=v.get("benchmark_score", 0),
                )
            )
            existing_version_ids.add(vid)
            restored_versions += 1
        await session.commit()
    logger.info(
        "Restore from backup=%s: models=%d versions=%d skipped=%d",
        backup_file,
        restored_models,
        restored_versions,
        skipped,
    )
    return {"models_restored": restored_models, "versions_restored": restored_versions, "skipped": skipped}
