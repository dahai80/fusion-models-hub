"""Model converter — converts models from various formats to MLX via fusion-mlx API.

Supports: HuggingFace safetensors, PyTorch, GGUF → MLX
All conversion goes through fusion-mlx HTTP API. Never imports torch/transformers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ModelConverter:
    """Converts models from various formats to Fusion-MLX (.mlx) format.

    All conversion operations call fusion-mlx's conversion API.
    """

    def __init__(self, mlx_url: str = "http://localhost:11434"):
        self.mlx_url = mlx_url.rstrip("/")

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
                resp = await client.post(f"{self.mlx_url}/v1/convert", json=request)
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

        output_path = str(path.parent / f"{path.stem}-{bits}bit.mlx")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{self.mlx_url}/v1/quantize", json={
                    "source_path": str(path),
                    "output_path": output_path,
                    "quant_bits": bits,
                })
                resp.raise_for_status()
                data = resp.json()
                return {
                    "status": "completed",
                    "output_path": output_path,
                    "quant_bits": bits,
                    "original_size_gb": data.get("original_size_gb", 0),
                    "converted_size_gb": data.get("converted_size_gb", 0),
                }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

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