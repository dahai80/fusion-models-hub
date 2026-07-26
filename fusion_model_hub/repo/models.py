"""Model data models — supports all model formats, with MLX as primary target."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..db.models import ModelFormat, ModelSource, ModelType, Quantization


@dataclass
class ModelInfo:
    """Model metadata — supports all formats, with MLX as primary target."""

    id: str
    name: str
    description: str = ""
    model_type: ModelType = ModelType.CHAT
    format: ModelFormat = ModelFormat.MLX
    quantization: Quantization = Quantization.Q4
    parameters: str = "7B"
    source: ModelSource = ModelSource.OFFICIAL

    # Fusion-MLX specific (when format=mlx)
    mlx_version: str = ""
    min_memory_gb: int = 8
    speed_tok_s: float = 0.0
    compatible_devices: list[str] = field(default_factory=lambda: ["M1", "M2", "M3", "M4", "M5"])

    # File info
    file_size_gb: float = 0.0
    file_hash: str = ""
    download_url: str = ""
    local_path: str = ""
    hf_repo: str = ""  # HuggingFace repo ID (e.g., "Qwen/Qwen2.5-7B")

    # Version
    version: str = "1.0.0"
    compatible_components: list[str] = field(default_factory=lambda: ["desktop", "agent", "kb", "bench"])
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type.value,
            "format": self.format.value,
            "quantization": self.quantization.value,
            "parameters": self.parameters,
            "source": self.source.value,
            "mlx_version": self.mlx_version,
            "min_memory_gb": self.min_memory_gb,
            "speed_tok_s": self.speed_tok_s,
            "compatible_devices": self.compatible_devices,
            "file_size_gb": self.file_size_gb,
            "file_hash": self.file_hash,
            "download_url": self.download_url,
            "local_path": self.local_path,
            "hf_repo": self.hf_repo,
            "version": self.version,
            "compatible_components": self.compatible_components,
            "tags": self.tags,
        }

    @property
    def is_mlx(self) -> bool:
        return self.format == ModelFormat.MLX

    @property
    def needs_conversion(self) -> bool:
        """Check if model needs conversion to MLX format."""
        return self.format != ModelFormat.MLX


@dataclass
class DownloadTask:
    """Download task tracking."""
    model_id: str
    url: str
    format: ModelFormat = ModelFormat.MLX
    file_size: int = 0
    downloaded: int = 0
    status: str = "pending"  # pending, downloading, verifying, converting, completed, failed
    hash: str = ""
    error: str = ""