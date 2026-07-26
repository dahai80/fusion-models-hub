import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from .base import StorageBackend

logger = logging.getLogger(__name__)

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB


class LocalStore(StorageBackend):
    """Local filesystem storage for model files with chunked upload and hash verification."""

    def __init__(self, data_dir: str = ""):
        if not data_dir:
            data_dir = str(Path.cwd() / "data")
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.uploads_dir = self.data_dir / "uploads"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def model_version_dir(self, model_id: str, version: str) -> Path:
        d = self.models_dir / model_id / version
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
        chunk_path.write_bytes(chunk_data)
        logger.info("Wrote chunk: upload=%s index=%d size=%d", upload_id, chunk_index, len(chunk_data))
        return chunk_path

    async def assemble_chunks(
        self, upload_id: str, target_dir: Path, filename: str, total_chunks: int,
    ) -> tuple[Path, str, int]:
        tmp_dir = self.upload_tmp_dir(upload_id)
        target_path = target_dir / filename
        hasher = hashlib.sha256()
        total_size = 0

        with open(target_path, "wb") as out:
            for i in range(total_chunks):
                chunk_path = tmp_dir / f"{i:06d}.part"
                if not chunk_path.exists():
                    raise FileNotFoundError(f"Missing chunk {i} for upload {upload_id}")
                data = chunk_path.read_bytes()
                out.write(data)
                hasher.update(data)
                total_size += len(data)

        file_hash = hasher.hexdigest()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(
            "Assembled upload: id=%s file=%s size=%d hash=%s",
            upload_id, filename, total_size, file_hash[:16],
        )
        return target_path, file_hash, total_size

    async def write_file(self, target_dir: Path, filename: str, data: bytes) -> tuple[Path, str, int]:
        target_path = target_dir / filename
        target_path.write_bytes(data)
        file_hash = hashlib.sha256(data).hexdigest()
        logger.info("Wrote file: %s size=%d", filename, len(data))
        return target_path, file_hash, len(data)

    def get_file(self, file_path: str) -> Path | None:
        p = Path(file_path)
        if p.exists():
            return p
        return None

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
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        result = h.hexdigest() == expected_hash.lower()
        if not result:
            logger.warning(
                "Hash mismatch: file=%s expected=%s actual=%s",
                file_path, expected_hash[:16], h.hexdigest()[:16],
            )
        return result

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
