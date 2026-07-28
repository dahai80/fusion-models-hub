from __future__ import annotations

import enum
from dataclasses import dataclass


class ChipGeneration(str, enum.Enum):
    M1 = "M1"
    M1_PRO = "M1_Pro"
    M1_MAX = "M1_Max"
    M1_ULTRA = "M1_Ultra"
    M2 = "M2"
    M2_PRO = "M2_Pro"
    M2_MAX = "M2_Max"
    M2_ULTRA = "M2_Ultra"
    M3 = "M3"
    M3_PRO = "M3_Pro"
    M3_MAX = "M3_Max"
    M4 = "M4"
    M4_PRO = "M4_Pro"
    M4_MAX = "M4_Max"
    M5 = "M5"
    M5_PRO = "M5_Pro"
    M5_MAX = "M5_Max"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class GPUProfile:
    name: str
    vendor: str
    vram_bytes: int
    vram_gb: float
    memory_bandwidth_gbps: float
    shared_memory: bool
    chip_generation: ChipGeneration


@dataclass(frozen=True)
class CPUProfile:
    name: str
    cores: int


@dataclass(frozen=True)
class HardwareProfile:
    gpu: GPUProfile | None
    cpu: CPUProfile
    ram_bytes: int
    ram_gb: float
    disk_free_bytes: int
    disk_free_gb: float
    os_name: str

    @property
    def total_memory_gb(self) -> float:
        return self.ram_gb

    @property
    def effective_vram_gb(self) -> float:
        if self.gpu:
            return self.gpu.vram_gb
        return self.ram_gb
