from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_WEIGHT_PROFILES = {
    "quality": {"hardware": 0.3, "quality": 0.4, "speed": 0.1, "popularity": 0.2},
    "balanced": {"hardware": 0.4, "quality": 0.2, "speed": 0.2, "popularity": 0.2},
    "speed": {"hardware": 0.2, "quality": 0.1, "speed": 0.5, "popularity": 0.2},
}

_QUANT_QUALITY = {
    "FP16": 100, "BF16": 100, "F32": 100,
    "Q8_0": 85, "Q6_K": 75, "Q5_K_M": 65, "Q5_0": 60,
    "Q4_K_M": 55, "Q4_0": 50, "Q3_K_M": 40, "Q2_K": 25, "IQ1_M": 15,
}


def score_hardware_fit(vram_required_gb: float, vram_available_gb: float, can_run: bool) -> float:
    if not can_run or vram_available_gb <= 0:
        return 0.0
    ratio = vram_required_gb / vram_available_gb
    if ratio <= 0.5:
        return 100.0
    if ratio <= 0.8:
        return 80.0 + 20.0 * (0.8 - ratio) / 0.3
    if ratio <= 1.0:
        return 40.0 + 40.0 * (1.0 - ratio) / 0.2
    return max(0.0, 40.0 * max(0, 1.0 - (ratio - 1.0)))


def score_quality(quant_type: str) -> float:
    return _QUANT_QUALITY.get(quant_type, 50.0)


def score_speed(tok_per_sec: float) -> float:
    if tok_per_sec <= 0:
        return 0.0
    return min(100.0, tok_per_sec / 0.5)


def score_popularity(download_count: int) -> float:
    if download_count <= 0:
        return 20.0
    if download_count >= 100000:
        return 100.0
    return 20.0 + 80.0 * min(download_count / 100000, 1.0)


def weighted_total(
    hardware: float,
    quality: float,
    speed: float,
    popularity: float,
    preference: str = "balanced",
) -> float:
    weights = _WEIGHT_PROFILES.get(preference, _WEIGHT_PROFILES["balanced"])
    return (
        weights["hardware"] * hardware
        + weights["quality"] * quality
        + weights["speed"] * speed
        + weights["popularity"] * popularity
    )


def build_reason(
    can_run: bool,
    hardware: float,
    quality: float,
    speed: float,
    preference: str,
) -> str:
    if not can_run:
        return "Insufficient VRAM for this model on current hardware"
    parts = []
    if hardware >= 80:
        parts.append("excellent hardware fit")
    elif hardware >= 50:
        parts.append("good hardware fit")
    else:
        parts.append("tight VRAM budget")

    if quality >= 80:
        parts.append("high quantization quality")
    if speed >= 70 and preference == "speed":
        parts.append("fast inference")
    elif speed < 20 and speed > 0:
        parts.append("slow inference expected")

    return "; ".join(parts) if parts else "suitable for current hardware"
