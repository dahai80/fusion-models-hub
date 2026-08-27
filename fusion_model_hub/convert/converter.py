"""Model converter — converts models from various formats to MLX via fusion-mlx API.

Supports: HuggingFace safetensors, PyTorch, GGUF → MLX
All conversion goes through fusion-mlx HTTP API. Never imports torch/transformers.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _hash_and_size(path: Path) -> tuple[str, int]:
    # MLX quantize output is a model directory (safetensors + config + ...).
    # Hash every regular file under it, sorted by relative path for a stable
    # digest; sum byte sizes. A plain file path is hashed directly. Returns
    # (sha256_hex, total_bytes). Used by the converter's provenance fallback
    # so a quantized ModelVersion gets a real hash + size instead of empty/0.
    from ..utils.hashing import compute_sha256

    if path.is_file():
        size = path.stat().st_size
        return compute_sha256(path), size
    if not path.is_dir():
        return "", 0
    h = hashlib.sha256()
    total = 0
    for f in sorted(path.rglob("*")):
        if f.is_file():
            total += f.stat().st_size
            h.update(compute_sha256(f).encode())
    return h.hexdigest(), total


class ModelConverter:
    """Converts models from various formats to Fusion-MLX (.mlx) format.

    All conversion operations call fusion-mlx's conversion API.
    """

    def __init__(self, mlx_url: str = "http://localhost:11434", api_key: str = ""):
        self.mlx_url = mlx_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        # Fusion-MLX /v1/* endpoints require Authorization: Bearer <key>. An
        # empty key omits the header so an unauthenticated MLX still works.
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def convert(self, source_path: str, source_format: str = "",
                      output_path: str = "", quant_bits: int = 4,
                      model_name: str = "", hf_repo: str = "") -> dict[str, Any]:
        """Convert a model to Fusion-MLX format via fusion-mlx API.

        Args:
            source_path: Local path to source model (for local files).
            source_format: Source format (auto-detect if empty).
            output_path: Output path for .mlx file.
            quant_bits: Quantization bits (4 or 8).
            model_name: Optional model name.
            hf_repo: HuggingFace repo ID (e.g., "Qwen/Qwen2.5-7B").

        Returns:
            Dict with conversion status.
        """
        # Detect format
        fmt = source_format or self._detect_format(source_path, hf_repo)

        # Build request
        request = {
            "source_path": source_path,
            "source_format": fmt,
            "output_path": output_path or str(
                Path.cwd() / f"{model_name or Path(source_path).stem}-{quant_bits}bit.mlx"
            ),
            "quant_bits": quant_bits,
            "model_name": model_name or Path(source_path).stem,
        }
        if hf_repo:
            request["hf_repo"] = hf_repo

        # Validate source exists
        if source_path and not Path(source_path).expanduser().resolve().exists():
            return {"status": "failed", "error": f"Source not found: {source_path}"}

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{self.mlx_url}/v1/convert", json=request, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                return {
                    "status": "completed",
                    "output_path": data.get("output_path", request["output_path"]),
                    "source_format": fmt,
                    "quant_bits": quant_bits,
                    "original_size_gb": data.get("original_size_gb", 0),
                    "converted_size_gb": data.get("converted_size_gb", 0),
                    "compression_ratio": data.get("compression_ratio", 0),
                    "compatible": data.get("compatible", True),
                }
        except httpx.HTTPStatusError as e:
            return {"status": "failed", "error": f"Conversion API error ({e.response.status_code})"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def convert_from_hf(self, hf_repo: str, quant_bits: int = 4,
                               output_path: str = "") -> dict[str, Any]:
        """Convert a HuggingFace model directly from repo ID."""
        return await self.convert(
            source_path="",
            source_format="huggingface",
            hf_repo=hf_repo,
            quant_bits=quant_bits,
            model_name=hf_repo.split("/")[-1],
            output_path=output_path,
        )

    async def quantize(self, mlx_path: str, bits: int = 4) -> dict[str, Any]:
        """Re-quantize an existing model to a different bit level."""
        path = Path(mlx_path).expanduser().resolve()
        if not path.exists():
            return {"status": "failed", "error": f"File not found: {path}"}

        # Upstream contract (fusion-mlx#646): /v1/quantize requires a `model`
        # field (HF repo / alias / local model path), NOT `source_path`. The
        # output_path must fall under an MLX-allowed directory
        # (~/.fusion-mlx/models, CWD, or HF cache) or MLX rejects it with 400.
        # MLX quantize is an async job: POST returns {job_id, status:queued};
        # poll GET /v1/quantize/jobs/{job_id} until status=="done".
        mlx_models_dir = Path.home() / ".fusion-mlx" / "models"
        output_path = str(mlx_models_dir / f"{path.stem}-{bits}bit.mlx")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{self.mlx_url}/v1/quantize", json={
                    "model": str(path),
                    "output_path": output_path,
                    "quant_bits": bits,
                }, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
                job_id = data.get("job_id", "")
                # Sync-style response (older MLX): no job_id means MLX returned
                # the result directly. Honor it so the contract is bidirectional.
                # A sync response carries an output_path and (optionally) a
                # status; treat a missing status as completed when an
                # output_path is present, since legacy MLX returned the result
                # dict without a status field.
                if not job_id:
                    job_data = data
                    if job_data.get("status") not in ("done", "completed") and job_data.get("output_path"):
                        job_data = {**job_data, "status": "completed"}
                else:
                    job_data = await self._poll_quantize_job(client, job_id)
                if job_data.get("status") not in ("done", "completed"):
                    return {
                        "status": "failed",
                        "error": job_data.get("error") or f"quantize job ended with status={job_data.get('status')!r}",
                    }
                result = {
                    "status": "completed",
                    "output_path": job_data.get("output_path", output_path),
                    "quant_bits": bits,
                    "original_size_gb": job_data.get("original_size_gb", 0),
                    "converted_size_gb": job_data.get("converted_size_gb", 0),
                }
                # P1-5: provenance repair. Before this, quantize returned no
                # file_hash/file_size, so the output ModelVersion was recorded
                # with empty hash + 0 bytes — corrupt provenance that broke
                # integrity checks, cache validation, and download resumes.
                # Prefer MLX-reported values; fall back to a local SHA256+size
                # pass on the output when MLX omits them. MLX quantize emits a
                # model DIRECTORY (weights + config + tokenizer), so the hash
                # covers every file under it, sorted by relative path for
                # determinism; size is the summed byte count.
                out = Path(result["output_path"])
                result["file_hash"] = job_data.get("file_hash", "")
                result["file_size"] = job_data.get("file_size", 0)
                if (not result["file_hash"] or not result["file_size"]) and out.exists():
                    import anyio

                    digest, size = await anyio.to_thread.run_sync(_hash_and_size, out)
                    if not result["file_hash"]:
                        result["file_hash"] = digest
                    if not result["file_size"]:
                        result["file_size"] = size
                    logger.info(
                        "Computed quantize output provenance: path=%s size=%d hash=%s",
                        out, size, digest[:16],
                    )
                return result
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _poll_quantize_job(
        self, client: httpx.AsyncClient, job_id: str,
        timeout_s: float = 600.0, interval_s: float = 1.0,
    ) -> dict[str, Any]:
        """Poll MLX /v1/quantize/jobs/{id} until the job is done or failed."""
        import time

        deadline = time.monotonic() + timeout_s
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            resp = await client.get(f"{self.mlx_url}/v1/quantize/jobs/{job_id}", headers=self._headers())
            if resp.status_code == 404:
                return {"status": "failed", "error": f"quantize job {job_id} not found"}
            resp.raise_for_status()
            last = resp.json()
            status = last.get("status", "")
            if status in ("done", "completed", "failed", "error"):
                return last
            await asyncio.sleep(interval_s)
        return {
            "status": "failed",
            "error": f"quantize job {job_id} timed out after {timeout_s}s",
            **last,
        }

    @staticmethod
    def _detect_format(source_path: str, hf_repo: str = "") -> str:
        """Detect model format from file path or repo."""
        if hf_repo:
            return "huggingface"
        if not source_path:
            return "unknown"
        ext = Path(source_path).suffix.lower()
        format_map = {
            ".safetensors": "huggingface",
            ".bin": "pytorch",
            ".gguf": "gguf",
            ".mlx": "mlx",
            ".pt": "pytorch",
            ".pth": "pytorch",
            ".onnx": "onnx",
        }
        return format_map.get(ext, "unknown")