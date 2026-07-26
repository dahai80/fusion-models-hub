import asyncio
import logging
from datetime import datetime, timezone

from ..convert.converter import ModelConverter
from ..db.crud import create_quantize_task, create_version, get_quantize_task, get_version, update_quantize_task, update_version, update_version_status
from ..db.models import ModelFormat, Quantization, TaskStatus, VersionStatus
from .config import Settings
from .deps import get_session_factory, get_settings

logger = logging.getLogger(__name__)

_running_tasks: dict[str, asyncio.Task] = {}


async def submit_quantize(
    source_version_id: str,
    target_format: str = "mlx",
    quant_bits: int = 4,
) -> str:
    session_factory = get_session_factory()
    settings = get_settings()
    async with session_factory() as session:
        task = await create_quantize_task(
            session,
            source_version_id=source_version_id,
            target_format=target_format,
            quant_bits=quant_bits,
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


async def _run_quantize(
    task_id: str,
    source_version_id: str,
    target_format: str,
    quant_bits: int,
    settings: Settings,
) -> None:
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            await update_quantize_task(
                session, task_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
            )

        source_ver = None
        async with session_factory() as session:
            source_ver = await get_version(session, source_version_id)

        if not source_ver:
            async with session_factory() as session:
                await update_quantize_task(
                    session, task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"Source version {source_version_id} not found",
                    completed_at=datetime.now(timezone.utc),
                )
            return

        quant_map = {2: Quantization.Q2, 4: Quantization.Q4, 6: Quantization.Q6, 8: Quantization.Q8}
        quant_enum = quant_map.get(quant_bits, Quantization.Q4)
        format_enum = ModelFormat(target_format)

        converter = ModelConverter(mlx_url=settings.mlx_url)
        result = await converter.quantize(
            mlx_path=source_ver.file_path,
            bits=quant_bits,
        )

        output_path = result.get("output_path", "")
        output_hash = result.get("file_hash", "")
        output_size = result.get("file_size", 0)

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
            async with session_factory() as session:
                await update_quantize_task(
                    session, task_id,
                    status=TaskStatus.COMPLETED,
                    output_version_id=new_ver.id,
                    completed_at=datetime.now(timezone.utc),
                )
            logger.info("Quantize task completed: id=%s output_ver=%s", task_id, new_ver.id)
        else:
            async with session_factory() as session:
                await update_quantize_task(
                    session, task_id,
                    status=TaskStatus.FAILED,
                    error_message="Failed to create output version",
                    completed_at=datetime.now(timezone.utc),
                )

    except Exception as e:
        logger.exception("Quantize task failed: id=%s", task_id)
        try:
            async with session_factory() as session:
                await update_quantize_task(
                    session, task_id,
                    status=TaskStatus.FAILED,
                    error_message=str(e),
                    completed_at=datetime.now(timezone.utc),
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
            "status": task.status.value,
            "output_version_id": task.output_version_id,
            "error_message": task.error_message,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "created_at": task.created_at.isoformat() if task.created_at else None,
        }


def list_running_tasks() -> list[dict]:
    return [
        {"id": tid, "name": t.get_name(), "done": t.done()}
        for tid, t in _running_tasks.items()
    ]
