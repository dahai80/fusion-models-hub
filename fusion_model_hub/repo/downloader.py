"""Model downloader — downloads and verifies Fusion-MLX models via HTTP."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Downloads Fusion-MLX models with resume support and hash verification."""

    def __init__(self, storage_dir: str = ""):
        if not storage_dir:
            storage_dir = str(Path.home() / "Library" / "Fusion" / "Models")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def download(self, url: str, model_id: str, expected_hash: str = "",
                       on_progress=None) -> dict[str, Any]:
        """Download a model file with resume support.

        Supports HTTP Range header for resuming partial downloads.
        If a .part file exists from a previous interrupted download,
        the download resumes from where it left off.

        Args:
            url: Download URL for the .mlx model file.
            model_id: Local model identifier.
            expected_hash: SHA256 hash for verification.
            on_progress: Optional callback(bytes_downloaded, total_bytes).

        Returns:
            Dict with status, path, and verification result.
        """
        import httpx
        file_path = self.storage_dir / f"{model_id}.mlx"
        temp_path = self.storage_dir / f"{model_id}.mlx.part"
        resume_offset = 0

        if temp_path.exists():
            resume_offset = temp_path.stat().st_size
            logger.info("Resuming download for %s from byte %d", model_id, resume_offset)

        try:
            headers = {}
            if resume_offset > 0:
                headers["Range"] = f"bytes={resume_offset}-"

            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client, \
                    client.stream("GET", url, headers=headers) as resp:
                    if resume_offset > 0 and resp.status_code not in (206, 200):
                        logger.warning("Server does not support resume, restarting from 0")
                        resume_offset = 0
                        temp_path.unlink(missing_ok=True)

                    if resp.status_code == 206:
                        total = int(resp.headers.get("content-range", "").split("/")[-1].strip())
                    else:
                        total = int(resp.headers.get("content-length", 0))
                        if resume_offset > 0:
                            resume_offset = 0
                            temp_path.unlink(missing_ok=True)

                    resp.raise_for_status()
                    downloaded = resume_offset

                    mode = "ab" if resume_offset > 0 else "wb"
                    with open(temp_path, mode) as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
                            downloaded += len(chunk)
                            if on_progress:
                                on_progress(downloaded, total)

            temp_path.rename(file_path)

            hash_ok = True
            if expected_hash:
                hash_ok = self._verify_hash(file_path, expected_hash)

            return {
                "status": "completed" if hash_ok else "hash_mismatch",
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "hash_verified": hash_ok,
                "resumed": resume_offset > 0,
            }

        except Exception as e:
            logger.error("Download failed for %s: %s (part file preserved for resume)", model_id, e)
            return {"status": "failed", "error": str(e)}

    def verify_local(self, file_path: str | Path, expected_hash: str) -> bool:
        """Verify a local model file against its expected hash."""
        return self._verify_hash(Path(file_path), expected_hash)

    @staticmethod
    def _verify_hash(file_path: Path, expected_hash: str) -> bool:
        """Verify SHA256 hash of a file."""
        try:
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest() == expected_hash.lower()
        except Exception as e:
            logger.error("Hash verification failed: %s", e)
            return False

    def get_storage_info(self) -> dict[str, Any]:
        """Get storage directory information."""
        total_size = 0
        file_count = 0
        if self.storage_dir.exists():
            for f in self.storage_dir.glob("*.mlx"):
                total_size += f.stat().st_size
                file_count += 1
        return {
            "path": str(self.storage_dir),
            "file_count": file_count,
            "total_size_gb": round(total_size / (1024**3), 2),
        }
