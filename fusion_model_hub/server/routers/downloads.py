import asyncio
import contextlib
import logging
import os
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...db import crud
from ..deps import SessionDep, SettingsDep, get_session_factory
from ..ssrf import validate_external_url

logger = logging.getLogger(__name__)
router = APIRouter(tags=["downloads"])

# Hold strong refs to background download tasks so CPython does not GC them
# mid-flight (asyncio.create_task only keeps a weak ref). Mirrors the quantize
# runner's _running_tasks pattern. Cleared via done-callback.
_running_downloads: set[asyncio.Task] = set()


class DownloadCreate(BaseModel):
    model_id: str
    source_url: str
    version_id: str = ""
    speed_limit_kbps: int = 0
    max_retries: int = 3
    expected_sha256: str = ""


@router.post("/downloads", status_code=201)
async def create_download(body: DownloadCreate, session: SessionDep, settings: SettingsDep):
    validate_external_url(body.source_url)

    m = await crud.get_model(session, body.model_id)
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")

    task = await crud.create_download_task(
        session,
        model_id=body.model_id,
        source_url=body.source_url,
        version_id=body.version_id,
        speed_limit_kbps=body.speed_limit_kbps,
        max_retries=body.max_retries,
        expected_sha256=body.expected_sha256,
    )

    _download_task = asyncio.create_task(_run_download(task.id, body.source_url, settings))
    _running_downloads.add(_download_task)
    _download_task.add_done_callback(_running_downloads.discard)

    logger.info("Download task created: id=%s model=%s url=%s", task.id, body.model_id, body.source_url)
    return {
        "task_id": task.id,
        "model_id": task.model_id,
        "status": task.status,
        "source_url": task.source_url,
    }


@router.get("/downloads")
async def list_downloads(
    session: SessionDep,
    model_id: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 20,
):
    tasks, total = await crud.list_download_tasks(
        session, model_id=model_id, status=status, page=page, page_size=page_size,
    )
    return {
        "tasks": [
            {
                "task_id": t.id,
                "model_id": t.model_id,
                "version_id": t.version_id,
                "source_url": t.source_url,
                "status": t.status,
                "progress_percent": t.progress_percent,
                "downloaded_bytes": t.downloaded_bytes,
                "total_bytes": t.total_bytes,
                "speed_limit_kbps": t.speed_limit_kbps,
                "retry_count": t.retry_count,
                "max_retries": t.max_retries,
                "error_message": t.error_message,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/downloads/{task_id}")
async def get_download(task_id: str, session: SessionDep):
    t = await crud.get_download_task(session, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Download task not found")
    return {
        "task_id": t.id,
        "model_id": t.model_id,
        "version_id": t.version_id,
        "source_url": t.source_url,
        "status": t.status,
        "progress_percent": t.progress_percent,
        "downloaded_bytes": t.downloaded_bytes,
        "total_bytes": t.total_bytes,
        "speed_limit_kbps": t.speed_limit_kbps,
        "retry_count": t.retry_count,
        "max_retries": t.max_retries,
        "error_message": t.error_message,
        "file_path": t.file_path,
        "file_hash": t.file_hash,
        "expected_sha256": t.expected_sha256,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
    }


@router.delete("/downloads/{task_id}")
async def cancel_download(task_id: str, session: SessionDep):
    t = await crud.get_download_task(session, task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Download task not found")
    if t.status in ("completed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in {t.status} state")
    await crud.update_download_task(session, task_id, status="cancelled")
    logger.info("Download task cancelled: id=%s", task_id)
    return {"task_id": task_id, "status": "cancelled"}


async def _run_download(task_id: str, source_url: str, settings):
    sf = get_session_factory()
    max_retries = 3
    async with sf() as session:
        t = await crud.get_download_task(session, task_id)
        if t:
            max_retries = t.max_retries

    for attempt in range(max_retries + 1):
        try:
            async with sf() as session:
                await crud.update_download_task(
                    session, task_id, status="downloading", retry_count=attempt,
                )

            downloaded = 0
            total = 0
            headers = {}
            expected_sha256 = ""

            async with sf() as session:
                t = await crud.get_download_task(session, task_id)
                if t and t.downloaded_bytes > 0:
                    headers["Range"] = f"bytes={t.downloaded_bytes}-"
                    downloaded = t.downloaded_bytes
                if t:
                    expected_sha256 = (t.expected_sha256 or "").lower()

            async with (
                httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client,
                client.stream("GET", source_url, headers=headers) as resp,
            ):
                    if resp.status_code not in (200, 206):
                        raise Exception(f"HTTP {resp.status_code}")

                    total = int(resp.headers.get("content-length", 0))
                    if resp.status_code == 206:
                        content_range = resp.headers.get("content-range", "")
                        if "/" in content_range:
                            total = int(content_range.split("/")[-1])

                    download_dir = os.path.join(settings.data_dir, "downloads")
                    os.makedirs(download_dir, exist_ok=True)
                    part_path = os.path.join(download_dir, f"{task_id}.part")
                    final_path = os.path.join(download_dir, f"{task_id}.bin")
                    write_mode = "ab" if downloaded > 0 else "wb"

                    chunk_size = 1024 * 1024
                    last_update = time.time()
                    # H6: compute SHA256 over the streamed bytes. Before, the
                    # /downloads path wrote the body to disk with ZERO integrity
                    # check — a corrupt or MITM'd download was persisted as
                    # "completed". Hash incrementally (resume appends, so hash
                    # continues over the part file's existing bytes on resume).
                    import hashlib
                    hasher = hashlib.sha256()

                    with open(part_path, write_mode) as fh:
                        async for chunk in resp.aiter_bytes(chunk_size):
                            fh.write(chunk)
                            hasher.update(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_update >= 1.0:
                                progress = (downloaded / total * 100) if total > 0 else 0
                                async with sf() as session:
                                    await crud.update_download_task(
                                        session, task_id,
                                        downloaded_bytes=downloaded,
                                        total_bytes=total,
                                        progress_percent=round(progress, 1),
                                    )
                                last_update = now

                    if total > 0 and downloaded < total:
                        raise Exception(
                            f"Incomplete download: {downloaded}/{total} bytes"
                        )
                    file_hash = hasher.hexdigest()
                    os.replace(part_path, final_path)

            # H6: if the caller supplied an expected hash, a mismatch is a hard
            # failure (corrupt/MITM'd bytes) — NOT a silent "completed". Integrity
            # failure is deterministic, so fail immediately (no retry loop).
            if expected_sha256 and file_hash.lower() != expected_sha256:
                with contextlib.suppress(OSError):
                    os.remove(final_path)
                msg = (
                    f"Download integrity check failed: sha256={file_hash} "
                    f"expected={expected_sha256}"
                )
                async with sf() as session:
                    await crud.update_download_task(
                        session, task_id,
                        status="failed",
                        error_message=msg[:500],
                        retry_count=attempt,
                    )
                logger.error("Download integrity failed (no retry): id=%s %s", task_id, msg)
                return

            async with sf() as session:
                await crud.update_download_task(
                    session, task_id,
                    status="completed",
                    downloaded_bytes=downloaded,
                    total_bytes=total,
                    progress_percent=100.0,
                    file_path=final_path,
                    file_hash=file_hash,
                )
            logger.info(
                "Download completed: id=%s bytes=%d path=%s sha256=%s",
                task_id, downloaded, final_path, file_hash,
            )
            return

        except asyncio.CancelledError:
            async with sf() as session:
                await crud.update_download_task(session, task_id, status="cancelled")
            logger.info("Download cancelled: id=%s", task_id)
            return

        except Exception as e:
            logger.warning(
                "Download attempt %d/%d failed for task %s: %s",
                attempt + 1, max_retries + 1, task_id, e,
            )
            if attempt >= max_retries:
                async with sf() as session:
                    await crud.update_download_task(
                        session, task_id,
                        status="failed",
                        error_message=str(e)[:500],
                        retry_count=attempt + 1,
                    )
                logger.error("Download permanently failed: id=%s error=%s", task_id, e)
                return
            await asyncio.sleep(2 ** attempt)
