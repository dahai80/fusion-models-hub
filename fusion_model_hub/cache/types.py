from __future__ import annotations

import enum
from dataclasses import dataclass, field


class CacheLevel(str, enum.Enum):
    RAW = "raw"
    CONVERTED = "converted"
    QUANTIZED = "quantized"


@dataclass
class CacheEntry:
    model_id: str
    level: CacheLevel
    path: str
    size_bytes: int
    sha256: str = ""
    quant_bits: int = 0
    mlx_version: str = ""
    source_version_id: str = ""
    created_at: float = 0.0
    last_accessed: float = 0.0
    ref_count: int = 0

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024**3), 2)


@dataclass
class CacheStats:
    total_entries: int = 0
    total_size_bytes: int = 0
    raw_count: int = 0
    converted_count: int = 0
    quantized_count: int = 0
    levels: dict[str, int] = field(default_factory=dict)

    @property
    def total_size_gb(self) -> float:
        return round(self.total_size_bytes / (1024**3), 2)
