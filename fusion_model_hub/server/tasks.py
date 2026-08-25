import asyncio
import logging
from datetime import UTC, datetime

from ..cache.types import CacheLevel
from ..convert.converter import ModelConverter
from ..db.crud import (
    claim_quantize_task,
    create_quantize_task,
    create_version,
    get_quantize_task,
    get_version,
    update_quantize_task,
)
from ..db.models import ModelFormat, Quantization, TaskStatus
from .config import Settings
from .deps import get_cache_manager, get_session_factory, get_settings

logger = logging.getLogger(__name__)

_running_tasks: dict[str, asyncio.Task] = {}

# R1: bound concurrent quantize executions. Each _run_quantize drives an MLX
# model load+quantize (hundreds of MB to GBs of unified memory); unbounded
# asyncio.create_task lets 50 concurrent POSTs load 50 models and OOM the
# process. The semaphore gates entries; the DB row is still created at PENDING
# immediately so the client can poll, and the worker flips it to RUNNING when
# it acquires the slot.
_QUANTIZE_CONCURRENCY = 4
_quantize_semaphore: asyncio.Semaphore | None = None


def _get_quantize_semaphore() -> asyncio.Semaphore:
    global _quantize_semaphore
    if _quantize_semaphore is None:
        _quantize_semaphore = asyncio.Semaphore(_QUANTIZE_CONCURRENCY)
    return _quantize_semaphore


async def submit_quantize(
    source_version_id: str,
    target_format: str = "mlx",
    quant_bits: int = 4,
    calibration_dataset: str = "",
) -> str:
    session_factory = get_session_factory()
    settings = get_settings()
    async with session_factory() as session:
        task = await create_quantize_task(
            session,
            source_version_id=source_version_id,
            target_format=target_format,
            quant_bits=quant_bits,
            calibration_dataset=calibration_dataset,
        )
        task_id = task.id

    atask = asyncio.create_task(
        _run_quantize(task_id, source_version_id, target_format, quant_bits, settings),
        name=f"quantize-{task_id}",
    )
    _running_tasks[task_id] = atask
    atask.add_done_callback(lambda t: _running_tasks.pop(task_id, None))
    logger.info("Submitted quantize task: id=%s", task_id)
    return task_id


async def resume_quantize(
    task_id: str,
    source_version_id: str,
    target_format: str,
    quant_bits: int,
) -> str | None:
    # R3: claim fencing on resume. Before spawning the task, atomically flip
    # PENDING->RUNNING via a conditional UPDATE. If a second hub process already
    # claimed it (rowcount==0), skip — never resume a task another worker owns.
    settings = get_settings()
    session_factory = get_session_factory()
    async with session_factory() as session:
        claimed = await claim_quantize_task(session, task_id)
    if not claimed:
        logger.warning("Skip resume: task already claimed by another worker: id=%s", task_id)
        return None
    atask = asyncio.create_task(
        _run_quantize(task_id, source_version_id, target_format, quant_bits, settings),
        name=f"quantize-{task_id}",
    )
    _running_tasks[task_id] = atask
    atask.add_done_callback(lambda t: _running_tasks.pop(task_id, None))
    logger.info("Resumed orphaned quantize task: id=%s", task_id)
    return task_id


async def _run_quantize(
    task_id: str,
    source_version_id: str,
    target_format: str,
    quant_bits: int,
    settings: Settings,
) -> None:
    session_factory = get_session_factory()
    # R1: gate the whole run on the concurrency semaphore so N concurrent
    # POSTs cannot load N models into MLX at once (unified-memory OOM). A
    # queued task waits here at PENDING; it flips to RUNNING only once it wins
    # a slot, so list_quantize_tasks reflects the real backlog.
    sem = _get_quantize_semaphore()
    try:
        async with sem:
            async with session_factory() as session:
                await update_quantize_task(
                    session,
                    task_id,
                    status=TaskStatus.RUNNING,
                    started_at=datetime.now(UTC),
                )

            source_ver = None
            async with session_factory() as session:
                source_ver = await get_version(session, source_version_id)

            if not source_ver:
                async with session_factory() as session:
                    await update_quantize_task(
                        session,
                        task_id,
                        status=TaskStatus.FAILED,
                        error_message=f"Source version {source_version_id} not found",
                        completed_at=datetime.now(UTC),
                    )
                return

            quant_map = {2: Quantization.Q2, 4: Quantization.Q4, 6: Quantization.Q6, 8: Quantization.Q8}
            quant_enum = quant_map.get(quant_bits, Quantization.Q4)
            format_enum = ModelFormat(target_format)

            converter = ModelConverter(mlx_url=settings.mlx_url)
            model_id = source_ver.model_id

            result = None
            cache_hit = False
            try:
                cache = get_cache_manager()
                if cache.has(model_id, CacheLevel.QUANTIZED, quant_bits, source_version_id=source_ver.id):
                    entry = cache.get(model_id, CacheLevel.QUANTIZED, quant_bits, source_version_id=source_ver.id)
                    if entry and entry.path:
                        logger.info(
                            "Cache hit for quantize: model=%s bits=%d ver=%s",
                            model_id, quant_bits, source_ver.id,
                        )
                        result = {
                            "status": "completed",
                            "output_path": entry.path,
                            "file_hash": entry.sha256,
                            "file_size": entry.size_bytes,
                        }
                        cache_hit = True
            except Exception:
                logger.exception("Cache lookup failed for quantize: model=%s", model_id)

            if not cache_hit:
                result = await converter.quantize(
                    mlx_path=source_ver.file_path,
                    bits=quant_bits,
                )
                out_path = result.get("output_path", "")
                if result.get("status") == "completed" and out_path:
                    try:
                        cache = get_cache_manager()
                        cache.put(
                            model_id=model_id,
                            level=CacheLevel.QUANTIZED,
                            source_path=out_path,
                            quant_bits=quant_bits,
                            source_version_id=source_ver.id,
                        )
                        logger.info(
                            "Cached quantize output: model=%s bits=%d ver=%s",
                            model_id, quant_bits, source_ver.id,
                        )
                    except Exception:
                        logger.exception("Cache put failed for quantize: model=%s", model_id)

            output_path = result.get("output_path", "")
            output_hash = result.get("file_hash", "")
            output_size = result.get("file_size", 0)

            result_status = result.get("status", "")
            if (result_status and result_status != "completed") or not output_path:
                async with session_factory() as session:
                    await update_quantize_task(
                        session,
                        task_id,
                        status=TaskStatus.FAILED,
                        error_message=(
                            f"Quantize produced no valid output: status={result_status!r} output_path={output_path!r}"
                        ),
                        completed_at=datetime.now(UTC),
                    )
                logger.error(
                    "Quantize produced no valid output: id=%s status=%r path=%r",
                    task_id,
                    result_status,
                    output_path,
                )
                return

            # H4: the output-version create and the task COMPLETED update MUST commit
            # in one transaction. Before, they were two separate sessions — a crash
            # between them left an orphan version plus a stuck RUNNING task forever.
            # Single session = both rows land or neither does; reconcile on startup
            # can still recover if even this commit fails.
            new_ver = None
            async with session_factory() as session:
                new_ver = await create_version(
                    session,
                    model_id=source_ver.model_id,
                    version=f"{source_ver.version}-{quant_bits}bit",
                    format=format_enum,
                    quantization=quant_enum,
                    file_path=output_path,
                    file_hash=output_hash,
                    file_size=output_size,
                    release_notes=f"Auto-quantized from {source_ver.version} ({quant_bits}-bit)",
                )
                if new_ver:
                    await update_quantize_task(
                        session,
                        task_id,
                        status=TaskStatus.COMPLETED,
                        output_version_id=new_ver.id,
                        completed_at=datetime.now(UTC),
                    )

            if new_ver:
                logger.info("Quantize task completed: id=%s output_ver=%s", task_id, new_ver.id)
                try:
                    from .routers.webhooks import dispatch_webhook_event

                    await dispatch_webhook_event("quantize.completed", {"id": task_id, "output_version_id": new_ver.id})
                except Exception:
                    logger.exception("Webhook dispatch failed for quantize.completed")
                try:
                    if settings.bench_auto_trigger and settings.bench_url:
                        import httpx

                        bench_payload = {
                            "suite": "general",
                            "model_id": source_ver.model_id,
                        }
                        async with httpx.AsyncClient(timeout=15.0) as client:
                            resp = await client.post(f"{settings.bench_url}/api/v1/tasks", json=bench_payload)
                            if resp.status_code in (200, 201, 202):
                                logger.info("Auto-triggered bench for model=%s after quantize", source_ver.model_id)
                            else:
                                logger.warning("Bench auto-trigger returned %d", resp.status_code)
                except Exception:
                    logger.exception("Bench auto-trigger failed for model=%s", source_ver.model_id)
                try:
                    threshold = settings.precision_loss_threshold
                    src_score = float(source_ver.benchmark_score or 0)
                    if src_score > 0 and new_ver:
                        async with session_factory() as session:
                            out_ver = await get_version(session, new_ver.id)
                            if out_ver:
                                out_score = float(out_ver.benchmark_score or 0)
                                if out_score > 0:
                                    loss_pct = (src_score - out_score) / src_score * 100
                                    if loss_pct > threshold:
                                        logger.warning(
                                            "Precision loss %.1f%% exceeds threshold %.1f%% for model=%s quant=%dbit",
                                            loss_pct,
                                            threshold,
                                            source_ver.model_id,
                                            quant_bits,
                                        )
                                        try:
                                            await dispatch_webhook_event(
                                                "quantize.precision_warning",
                                                {
                                                    "model_id": source_ver.model_id,
                                                    "source_version_id": source_ver.id,
                                                    "output_version_id": new_ver.id,
                                                    "loss_percent": round(loss_pct, 2),
                                                    "threshold": threshold,
                                                    "quant_bits": quant_bits,
                                                },
                                            )
                                        except Exception:
                                            logger.exception("Precision warning webhook dispatch failed")
                except Exception:
                    logger.exception("Precision loss check failed for model=%s", source_ver.model_id)
            else:
                async with session_factory() as session:
                    await update_quantize_task(
                        session,
                        task_id,
                        status=TaskStatus.FAILED,
                        error_message="Failed to create output version",
                        completed_at=datetime.now(UTC),
                    )

    except Exception as e:
        logger.exception("Quantize task failed: id=%s", task_id)
        try:
            from .routers.webhooks import dispatch_webhook_event

            await dispatch_webhook_event("quantize.failed", {"id": task_id, "error": str(e)})
        except Exception:
            logger.warning("quantize.failed webhook dispatch failed: id=%s", task_id, exc_info=True)
        try:
            async with session_factory() as session:
                await update_quantize_task(
                    session,
                    task_id,
                    status=TaskStatus.FAILED,
                    error_message=str(e),
                    completed_at=datetime.now(UTC),
                )
        except Exception:
            logger.exception("Failed to update task status on error: id=%s", task_id)


async def get_task_status(task_id: str) -> dict | None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        task = await get_quantize_task(session, task_id)
        if not task:
            return None
        return {
            "id": task.id,
            "source_version_id": task.source_version_id,
            "target_format": task.target_format,
            "quant_bits": task.quant_bits,
            "calibration_dataset": task.calibration_dataset,
            "status": task.status.value,
            "output_version_id": task.output_version_id,
            "error_message": task.error_message,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "created_at": task.created_at.isoformat() if task.created_at else None,
        }


def list_running_tasks() -> list[dict]:
    return [{"id": tid, "name": t.get_name(), "done": t.done()} for tid, t in _running_tasks.items()]
