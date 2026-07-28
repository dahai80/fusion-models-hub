from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ...hardware.detector import HardwareDetector
from ..deps import SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hardware", tags=["hardware"])

_detector: HardwareDetector | None = None


def _get_detector(settings: SettingsDep) -> HardwareDetector:
    global _detector
    if _detector is None or _detector.mlx_url != settings.mlx_url:
        _detector = HardwareDetector(settings.mlx_url)
    return _detector


@router.get("")
async def get_hardware_info(settings: SettingsDep):
    detector = _get_detector(settings)
    try:
        profile = await detector.detect()
    except Exception as e:
        logger.error("Hardware detection failed: %s", e)
        raise HTTPException(status_code=503, detail="Hardware detection unavailable — is Fusion-MLX running?")

    gpu_info = None
    if profile.gpu:
        gpu_info = {
            "name": profile.gpu.name,
            "vendor": profile.gpu.vendor,
            "vram_bytes": profile.gpu.vram_bytes,
            "vram_gb": round(profile.gpu.vram_gb, 2),
            "memory_bandwidth_gbps": profile.gpu.memory_bandwidth_gbps,
            "shared_memory": profile.gpu.shared_memory,
            "chip_generation": profile.gpu.chip_generation.value,
        }

    return {
        "gpu": gpu_info,
        "cpu": {
            "name": profile.cpu.name,
            "cores": profile.cpu.cores,
        },
        "ram": {
            "total_bytes": profile.ram_bytes,
            "total_gb": round(profile.ram_gb, 2),
        },
        "disk": {
            "free_bytes": profile.disk_free_bytes,
            "free_gb": round(profile.disk_free_gb, 2),
        },
        "os": profile.os_name,
    }


@router.post("/refresh")
async def refresh_hardware_info(settings: SettingsDep):
    detector = _get_detector(settings)
    detector.invalidate_cache()
    try:
        profile = await detector.detect()
    except Exception as e:
        logger.error("Hardware refresh failed: %s", e)
        raise HTTPException(status_code=503, detail="Hardware detection unavailable")
    return {"status": "refreshed", "chip": profile.gpu.chip_generation.value if profile.gpu else "Unknown"}
