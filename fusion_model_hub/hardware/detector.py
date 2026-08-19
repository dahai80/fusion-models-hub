from __future__ import annotations

import logging
import re
import time

import httpx

from .types import ChipGeneration, CPUProfile, GPUProfile, HardwareProfile

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # 5 minutes

_CHIP_PATTERNS = [
    (re.compile(r"M5\s+Max", re.I), ChipGeneration.M5_MAX),
    (re.compile(r"M5\s+Pro", re.I), ChipGeneration.M5_PRO),
    (re.compile(r"M5(?!\s)", re.I), ChipGeneration.M5),
    (re.compile(r"M4\s+Max", re.I), ChipGeneration.M4_MAX),
    (re.compile(r"M4\s+Pro", re.I), ChipGeneration.M4_PRO),
    (re.compile(r"M4(?!\s)", re.I), ChipGeneration.M4),
    (re.compile(r"M3\s+Max", re.I), ChipGeneration.M3_MAX),
    (re.compile(r"M3\s+Pro", re.I), ChipGeneration.M3_PRO),
    (re.compile(r"M3(?!\s)", re.I), ChipGeneration.M3),
    (re.compile(r"M2\s+Ultra", re.I), ChipGeneration.M2_ULTRA),
    (re.compile(r"M2\s+Max", re.I), ChipGeneration.M2_MAX),
    (re.compile(r"M2\s+Pro", re.I), ChipGeneration.M2_PRO),
    (re.compile(r"M2(?!\s)", re.I), ChipGeneration.M2),
    (re.compile(r"M1\s+Ultra", re.I), ChipGeneration.M1_ULTRA),
    (re.compile(r"M1\s+Max", re.I), ChipGeneration.M1_MAX),
    (re.compile(r"M1\s+Pro", re.I), ChipGeneration.M1_PRO),
    (re.compile(r"M1(?!\s)", re.I), ChipGeneration.M1),
]


def _parse_chip_generation(gpu_name: str) -> ChipGeneration:
    for pat, gen in _CHIP_PATTERNS:
        if pat.search(gpu_name):
            return gen
    return ChipGeneration.UNKNOWN


class HardwareDetector:
    def __init__(self, mlx_url: str = "http://localhost:11434"):
        self.mlx_url = mlx_url.rstrip("/")
        self._cache: HardwareProfile | None = None
        self._cache_time: float = 0

    async def detect(self) -> HardwareProfile:
        now = time.time()
        if self._cache and (now - self._cache_time) < _CACHE_TTL:
            logger.debug("Using cached hardware profile")
            return self._cache

        try:
            profile = await self._fetch_from_mlx()
            self._cache = profile
            self._cache_time = now
            return profile
        except Exception as e:
            logger.error("Hardware detection failed: %s", e)
            if self._cache:
                logger.warning("Returning stale cached profile")
                return self._cache
            return self._fallback_profile()

    async def _fetch_from_mlx(self) -> HardwareProfile:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.mlx_url}/v1/hardware")
            resp.raise_for_status()
            data = resp.json()

        gpu_data = data.get("gpu")
        gpu = None
        if gpu_data:
            gpu_name = gpu_data.get("name", "Unknown")
            gpu = GPUProfile(
                name=gpu_name,
                vendor=gpu_data.get("vendor", "Apple"),
                vram_bytes=gpu_data.get("vram_bytes", 0),
                vram_gb=gpu_data.get("vram_bytes", 0) / 1e9,
                memory_bandwidth_gbps=gpu_data.get("memory_bandwidth_gbps", 0),
                shared_memory=gpu_data.get("shared_memory", True),
                chip_generation=_parse_chip_generation(gpu_name),
            )

        cpu_data = data.get("cpu", {})
        cpu = CPUProfile(
            name=cpu_data.get("name", "Unknown"),
            cores=cpu_data.get("cores", 0),
        )

        ram_data = data.get("ram", {})
        disk_data = data.get("disk", {})

        profile = HardwareProfile(
            gpu=gpu,
            cpu=cpu,
            ram_bytes=ram_data.get("total_bytes", 0),
            ram_gb=ram_data.get("total_gb", 0),
            disk_free_bytes=disk_data.get("free_bytes", 0),
            disk_free_gb=disk_data.get("free_gb", 0),
            os_name=data.get("os", "macos"),
        )
        logger.info("Hardware detected: chip=%s vram=%.1fGB ram=%.1fGB",
                     profile.gpu.chip_generation.value if profile.gpu else "N/A",
                     profile.effective_vram_gb, profile.ram_gb)
        return profile

    @staticmethod
    def _fallback_profile() -> HardwareProfile:
        logger.warning("Using fallback hardware profile (unknown hardware)")
        return HardwareProfile(
            gpu=None,
            cpu=CPUProfile(name="Unknown", cores=0),
            ram_bytes=0,
            ram_gb=0.0,
            disk_free_bytes=0,
            disk_free_gb=0.0,
            os_name="unknown",
        )

    def invalidate_cache(self) -> None:
        self._cache = None
        self._cache_time = 0
