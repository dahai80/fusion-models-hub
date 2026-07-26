"""Fusion-Model-Hub — Unified model repository and manager for the Fusion-MLX ecosystem.

All model inference, conversion, and verification goes through fusion-mlx HTTP API.
Never imports MLX, mlx-lm, torch, or transformers directly.
"""

from .api.base_binding import FusionMLXBase
from .convert.converter import ModelConverter
from .db.models import ModelFormat, ModelSource, ModelType, Quantization
from .manage.manager import LocalModelManager
from .repo.models import ModelInfo
from .repo.registry import ModelRegistry

__all__ = [
    "FusionMLXBase",
    "LocalModelManager",
    "ModelConverter",
    "ModelFormat",
    "ModelInfo",
    "ModelRegistry",
    "ModelSource",
    "ModelType",
    "Quantization",
]
