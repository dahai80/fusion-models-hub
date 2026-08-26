import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import anyio

from .base import StorageBackend

logger = logging.getLogger(__name__)

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB


class LocalStore(StorageBackend):
    """Local filesystem storage for model files with chunked upload and hash verification."""

    def __init__(self, data_dir: str = ""):
        if not data_dir:
            data_dir = str(Path.cwd() / "data")
        self.data_dir = Path(data_dir)
        self._models_dir = self.data_dir / "models"
        self.uploads_dir = self.data_dir / "uploads"
        self.lfs_dir = self.data_dir / "lfs"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.lfs_dir.mkdir(parents=True, exist_ok=True)

    # E-R3: models_dir is now an abstract property on StorageBackend so the
    # contract is explicit and MinioStore cannot silently lack it. Expose the
    # private field set in __init__ via a property to satisfy the ABC without
    # changing any caller (they all read store.models_dir as an attribute).
    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def model_version_dir(self, model_id: str, version: str) -> Path:
        d = self._models_dir / model_id / version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def upload_tmp_dir(self, upload_id: str) -> Path:
        d = self.uploads_dir / upload_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def write_chunk(
        self, upload_id: str, chunk_index: int, chunk_data: bytes,
    ) -> Path:
        tmp_dir = self.upload_tmp_dir(upload_id)
        chunk_path = tmp_dir / f"{chunk_index:06d}.part"
        # P1-7: the disk write blocks the event loop; offload it.
        await anyio.to_thread.run_sync(chunk_path.write_bytes, chunk_data)
        logger.info("Wrote chunk: upload=%s index=%d size=%d", upload_id, chunk_index, len(chunk_data))
        return chunk_path

    def _assemble_chunks_sync(
        self, upload_id: str, target_dir: Path, filename: str, total_chunks: int,
    ) -> tuple[Path, str, int]:
        # E-D3: assemble to a side temp file, fsync, then atomic os.replace into
        # place. A crash mid-assemble leaves the old target untouched instead of a
        # truncated/corrupt file at the final path. Chunk tmp is cleaned in finally
        # so a failed/aborted upload does not leak .part files forever.
        # P1-11 (defense-in-depth): reduce filename to a bare basename so a
        # traversal value (../etc/x) cannot escape target_dir even if a caller
        # bypasses the router-level check.
        filename = os.path.basename(filename)
        if not filename or filename in (".", ".."):
            raise ValueError("Invalid filename")
        tmp_dir = self.upload_tmp_dir(upload_id)
        target_path = target_dir / filename
        staging_path = target_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        hasher = hashlib.sha256()
        total_size = 0
        try:
            with open(staging_path, "wb") as out:
                for i in range(total_chunks):
                    chunk_path = tmp_dir / f"{i:06d}.part"
                    if not chunk_path.exists():
                        raise FileNotFoundError(f"Missing chunk {i} for upload {upload_id}")
                    data = chunk_path.read_bytes()
                    out.write(data)
                    hasher.update(data)
                    total_size += len(data)
                out.flush()
                os.fsync(out.fileno())
            os.replace(staging_path, target_path)
        finally:
            if staging_path.exists():
                staging_path.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        file_hash = hasher.hexdigest()
        logger.info(
            "Assembled upload: id=%s file=%s size=%d hash=%s",
            upload_id, filename, total_size, file_hash[:16],
        )
        return target_path, file_hash, total_size

    async def assemble_chunks(
        self, upload_id: str, target_dir: Path, filename: str, total_chunks: int,
    ) -> tuple[Path, str, int]:
        # P1-7: the multi-chunk read+write+fsync blocks the event loop for the
        # whole upload; run the blocking assembly on a worker thread.
        return await anyio.to_thread.run_sync(
            self._assemble_chunks_sync, upload_id, target_dir, filename, total_chunks,
        )

    def _write_file_sync(self, target_dir: Path, filename: str, data: bytes) -> tuple[Path, str, int]:
        # E-D3: atomic write via staging file + os.replace.
        # P1-11 (defense-in-depth): bare basename so a traversal filename cannot
        # escape target_dir.
        filename = os.path.basename(filename)
        if not filename or filename in (".", ".."):
            raise ValueError("Invalid filename")
        target_path = target_dir / filename
        staging_path = target_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        try:
            with open(staging_path, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            os.replace(staging_path, target_path)
        finally:
            if staging_path.exists():
                staging_path.unlink(missing_ok=True)
        file_hash = hashlib.sha256(data).hexdigest()
        logger.info("Wrote file: %s size=%d", filename, len(data))
        return target_path, file_hash, len(data)

    async def write_file(self, target_dir: Path, filename: str, data: bytes) -> tuple[Path, str, int]:
        # P1-7: offload the blocking file write to a worker thread.
        return await anyio.to_thread.run_sync(self._write_file_sync, target_dir, filename, data)

    def get_file(self, file_path: str) -> Path | None:
        p = Path(file_path)
        if p.exists():
            return p
        return None

    def put_lfs_object(self, oid: str, data: bytes) -> Path:
        # FR-027 / P1-2: store a Git LFS object keyed by oid (content SHA256).
        # Atomic write via staging file + os.replace; oid is sanitized to a
        # bare filename so a malicious oid cannot escape lfs_dir.
        safe_oid = os.path.basename(oid)
        if not safe_oid or safe_oid != oid:
            raise ValueError("Invalid LFS oid")
        target = self.lfs_dir / safe_oid
        staging = self.lfs_dir / f".{safe_oid}.{uuid.uuid4().hex}.tmp"
        try:
            with open(staging, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            os.replace(staging, target)
        finally:
            if staging.exists():
                staging.unlink(missing_ok=True)
        logger.info("Stored LFS object: oid=%s size=%d", safe_oid[:16], len(data))
        return target

    def get_lfs_object(self, oid: str) -> Path | None:
        safe_oid = os.path.basename(oid)
        if not safe_oid or safe_oid != oid:
            return None
        p = self.lfs_dir / safe_oid
        return p if p.exists() else None

    def is_path_within_store(self, file_path: Path) -> bool:
        try:
            resolved = file_path.resolve()
            models_resolved = self.models_dir.resolve()
            return str(resolved).startswith(str(models_resolved))
        except (OSError, ValueError):
            return False

    def delete_version_files(self, model_id: str, version: str) -> bool:
        version_dir = self.models_dir / model_id / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            logger.info("Deleted version files: model=%s version=%s", model_id, version)
            return True
        return False

    def delete_model_files(self, model_id: str) -> bool:
        model_dir = self.models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info("Deleted model files: model=%s", model_id)
            return True
        return False

    @staticmethod
    def verify_hash(file_path: Path, expected_hash: str) -> bool:
        # E-E8: delegate to the shared utils helper (which logs mismatches).
        from ..utils.hashing import verify_sha256

        return verify_sha256(file_path, expected_hash)

    def get_storage_stats(self) -> dict[str, Any]:
        total_size = 0
        file_count = 0
        model_count = 0
        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    model_count += 1
                    for f in model_dir.rglob("*"):
                        if f.is_file():
                            total_size += f.stat().st_size
                            file_count += 1
        return {
            "path": str(self.models_dir),
            "model_count": model_count,
            "file_count": file_count,
            "total_size_gb": round(total_size / (1024**3), 2),
        }
