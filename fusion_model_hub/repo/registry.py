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
    def list(cls, model_type: str = "", model_format: str = "",
             quant: str = "", device: str = "", search: str = "",
             mlx_only: bool = False) -> list[dict[str, Any]]:
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
            # MLX format (native)
            ModelInfo(id="qwen3.5-9b-q4", name="Qwen3.5-9B (MLX 4bit)",
                      description="Alibaba Qwen3.5 9B, native MLX 4bit",
                      model_type=ModelType.CHAT, format=ModelFormat.MLX,
                      quantization=Quantization.Q4, parameters="9B",
                      speed_tok_s=35.0, min_memory_gb=8, file_size_gb=5.2,
                      mlx_version=">=0.5.0",
                      download_url="https://huggingface.co/mlx-community/Qwen3.5-9B-4bit"),
            ModelInfo(id="qwen3.5-9b-q8", name="Qwen3.5-9B (MLX 8bit)",
                      description="Alibaba Qwen3.5 9B, native MLX 8bit",
                      model_type=ModelType.CHAT, format=ModelFormat.MLX,
                      quantization=Quantization.Q8, parameters="9B",
                      speed_tok_s=25.0, min_memory_gb=16, file_size_gb=9.8,
                      mlx_version=">=0.5.0",
                      download_url="https://huggingface.co/mlx-community/Qwen3.5-9B-8bit"),
            # GGUF format
            ModelInfo(id="qwen3.5-9b-gguf-q4", name="Qwen3.5-9B (GGUF Q4_K_M)",
                      description="Qwen3.5 9B in GGUF format, compatible with Ollama/llama.cpp",
                      model_type=ModelType.CHAT, format=ModelFormat.GGUF,
                      quantization=Quantization.Q4, parameters="9B",
                      min_memory_gb=8, file_size_gb=5.5,
                      hf_repo="Qwen/Qwen2.5-7B-GGUF"),
            # HuggingFace format
            ModelInfo(id="qwen3.5-9b-hf", name="Qwen3.5-9B (HuggingFace)",
                      description="Qwen3.5 9B original HuggingFace format, needs conversion to MLX",
                      model_type=ModelType.CHAT, format=ModelFormat.SAFETENSORS,
                      quantization=Quantization.NONE, parameters="9B",
                      source=ModelSource.HUGGINGFACE, min_memory_gb=32, file_size_gb=18,
                      hf_repo="Qwen/Qwen3.5-9B"),
            # Embedding
            ModelInfo(id="bge-m3-embedding", name="BGE-M3 Embedding (MLX)",
                      description="BAAI BGE-M3, MLX format for Fusion-KB",
                      model_type=ModelType.EMBEDDING, format=ModelFormat.MLX,
                      quantization=Quantization.Q8, parameters="568M",
                      speed_tok_s=500.0, min_memory_gb=2, file_size_gb=1.1,
                      mlx_version=">=0.5.0"),
            # Multimodal
            ModelInfo(id="qwen-vl-7b-q4", name="Qwen2-VL-7B (MLX 4bit)",
                      description="Qwen2 Vision-Language 7B, MLX 4bit",
                      model_type=ModelType.MULTIMODAL, format=ModelFormat.MLX,
                      quantization=Quantization.Q4, parameters="7B",
                      speed_tok_s=30.0, min_memory_gb=8, file_size_gb=4.0,
                      mlx_version=">=0.5.0"),
            # DeepSeek MLX
            ModelInfo(id="deepseek-v4-flash-q4", name="DeepSeek-V4-Flash (MLX 4bit)",
                      description="DeepSeek V4 Flash 27B, MLX 4bit",
                      model_type=ModelType.CHAT, format=ModelFormat.MLX,
                      quantization=Quantization.Q4, parameters="27B",
                      speed_tok_s=18.0, min_memory_gb=16, file_size_gb=15.0,
                      mlx_version=">=0.5.0"),
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