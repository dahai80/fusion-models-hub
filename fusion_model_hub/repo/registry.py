"""Model registry — supports all model formats with MLX as primary target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ModelFormat, ModelInfo, ModelSource, ModelType, Quantization


class ModelRegistry:
    """Registry of models in all formats. MLX-format models are preferred."""

    _models: dict[str, ModelInfo] = {}

    @classmethod
    def register(cls, model: ModelInfo) -> None:
        if not model.id:
            raise ValueError("Model must have an id")
        cls._models[model.id] = model

    @classmethod
    def get(cls, model_id: str) -> ModelInfo | None:
        return cls._models.get(model_id)

    @classmethod
    def list(
        cls,
        model_type: str = "",
        model_format: str = "",
        quant: str = "",
        device: str = "",
        search: str = "",
        mlx_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List models with optional filters."""
        results = []
        for m in cls._models.values():
            if mlx_only and not m.is_mlx:
                continue
            if model_type and m.model_type.value != model_type:
                continue
            if model_format and m.format.value != model_format:
                continue
            if quant and m.quantization.value != quant:
                continue
            if device and device not in m.compatible_devices:
                continue
            if search:
                q = search.lower()
                if q not in m.name.lower() and q not in m.description.lower() and q not in m.id.lower():
                    continue
            results.append(m.to_dict())
        return results

    @classmethod
    def register_defaults(cls) -> None:
        """Register default models across formats."""
        defaults = [
            # MLX format (native) — real mlx-community repos. Prior defaults
            # listed "Qwen3.5" (never released) and "deepseek-v4-flash"
            # (fabricated) with HF 404 download_urls and invented tok/s numbers,
            # so a user who ran `fmh download` hit a 404 and the recommend
            # engine ranked nonexistent models. Replaced with real, fetchable
            # models; speeds are representative Apple-Silicon ballpark, not
            # measured claims.
            ModelInfo(
                id="qwen2.5-7b-instruct-mlx-4bit",
                name="Qwen2.5-7B-Instruct (MLX 4bit)",
                description="Alibaba Qwen2.5 7B Instruct, native MLX 4bit",
                model_type=ModelType.CHAT,
                format=ModelFormat.MLX,
                quantization=Quantization.Q4,
                parameters="7B",
                speed_tok_s=35.0,
                min_memory_gb=8,
                file_size_gb=4.4,
                mlx_version=">=0.5.0",
                download_url="https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit",
            ),
            ModelInfo(
                id="qwen2.5-7b-instruct-mlx-8bit",
                name="Qwen2.5-7B-Instruct (MLX 8bit)",
                description="Alibaba Qwen2.5 7B Instruct, native MLX 8bit",
                model_type=ModelType.CHAT,
                format=ModelFormat.MLX,
                quantization=Quantization.Q8,
                parameters="7B",
                speed_tok_s=25.0,
                min_memory_gb=12,
                file_size_gb=8.1,
                mlx_version=">=0.5.0",
                download_url="https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-8bit",
            ),
            # GGUF format
            ModelInfo(
                id="qwen2.5-7b-instruct-gguf-q4",
                name="Qwen2.5-7B-Instruct (GGUF Q4_K_M)",
                description="Qwen2.5 7B Instruct in GGUF format, compatible with Ollama/llama.cpp",
                model_type=ModelType.CHAT,
                format=ModelFormat.GGUF,
                quantization=Quantization.Q4,
                parameters="7B",
                min_memory_gb=8,
                file_size_gb=4.7,
                hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
            ),
            # HuggingFace format
            ModelInfo(
                id="qwen2.5-7b-instruct-hf",
                name="Qwen2.5-7B-Instruct (HuggingFace)",
                description="Qwen2.5 7B Instruct original HuggingFace format, needs conversion to MLX",
                model_type=ModelType.CHAT,
                format=ModelFormat.SAFETENSORS,
                quantization=Quantization.NONE,
                parameters="7B",
                source=ModelSource.HUGGINGFACE,
                min_memory_gb=16,
                file_size_gb=15.0,
                hf_repo="Qwen/Qwen2.5-7B-Instruct",
            ),
            # Embedding
            ModelInfo(
                id="bge-m3-embedding",
                name="BGE-M3 Embedding (MLX)",
                description="BAAI BGE-M3, MLX format for Fusion-KB",
                model_type=ModelType.EMBEDDING,
                format=ModelFormat.MLX,
                quantization=Quantization.Q8,
                parameters="568M",
                speed_tok_s=500.0,
                min_memory_gb=2,
                file_size_gb=1.1,
                mlx_version=">=0.5.0",
                download_url="https://huggingface.co/mlx-community/bge-m3-mlx",
            ),
            # Multimodal
            ModelInfo(
                id="qwen2-vl-7b-instruct-mlx-4bit",
                name="Qwen2-VL-7B-Instruct (MLX 4bit)",
                description="Qwen2 Vision-Language 7B Instruct, MLX 4bit",
                model_type=ModelType.MULTIMODAL,
                format=ModelFormat.MLX,
                quantization=Quantization.Q4,
                parameters="7B",
                speed_tok_s=30.0,
                min_memory_gb=8,
                file_size_gb=4.0,
                mlx_version=">=0.5.0",
                download_url="https://huggingface.co/mlx-community/Qwen2-VL-7B-Instruct-4bit",
            ),
            # Phi-3.5 mini MLX (replaces fabricated DeepSeek-V4-Flash)
            ModelInfo(
                id="phi-3.5-mini-instruct-mlx-4bit",
                name="Phi-3.5-Mini-Instruct (MLX 4bit)",
                description="Microsoft Phi-3.5 Mini 3.8B Instruct, MLX 4bit",
                model_type=ModelType.CHAT,
                format=ModelFormat.MLX,
                quantization=Quantization.Q4,
                parameters="3.8B",
                speed_tok_s=45.0,
                min_memory_gb=4,
                file_size_gb=2.2,
                mlx_version=">=0.5.0",
                download_url="https://huggingface.co/mlx-community/Phi-3.5-mini-instruct-4bit",
            ),
        ]
        for m in defaults:
            cls.register(m)

    @classmethod
    def count(cls) -> int:
        return len(cls._models)

    @classmethod
    def load_from_json(cls, path: str | Path) -> int:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        count = 0
        for item in data:
            # Convert string format fields to enums
            if "model_type" in item:
                item["model_type"] = ModelType(item["model_type"])
            if "format" in item:
                item["format"] = ModelFormat(item["format"])
            if "quantization" in item:
                item["quantization"] = Quantization(item["quantization"])
            if "source" in item:
                item["source"] = ModelSource(item["source"])
            cls.register(ModelInfo(**item))
            count += 1
        return count
