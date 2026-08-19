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
    gpu_cores = 0
    chip_name = "Unknown"
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
        chip_name = profile.gpu.name
        for tok in chip_name.split():
            if tok.isdigit():
                gpu_cores = int(tok)
                break

    cpu_info = {
        "name": profile.cpu.name,
        "cores": profile.cpu.cores,
    }
    ram_info = {
        "total_bytes": profile.ram_bytes,
        "total_gb": round(profile.ram_gb, 2),
    }
    disk_info = {
        "free_bytes": profile.disk_free_bytes,
        "free_gb": round(profile.disk_free_gb, 2),
    }

    logger.info(
        "Hardware detected: chip=%s cpu_cores=%d gpu_cores=%d ram_gb=%.2f disk_free_gb=%.2f",
        chip_name, profile.cpu.cores, gpu_cores, profile.ram_gb, profile.disk_free_gb,
    )

    return {
        # nested form kept for recommend/adapt internal callers
        "gpu": gpu_info,
        "cpu": cpu_info,
        "ram": ram_info,
        "disk": disk_info,
        "os": profile.os_name,
        # flat form for fusion-studio HubHardwareResponse (all keys optional there)
        "chip": chip_name,
        "cpuCores": profile.cpu.cores,
        "gpuCores": gpu_cores,
        "memoryGB": round(profile.ram_gb, 2),
        "diskFree": round(profile.disk_free_gb, 2),
        "metalSupport": True,
        "aneSupport": True,
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
