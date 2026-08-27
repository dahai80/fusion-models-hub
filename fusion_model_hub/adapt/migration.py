from __future__ import annotations

import logging

from ..hardware.types import HardwareProfile
from .types import AdaptationLevel, MigrationPlan, QuantizeSuggestion

logger = logging.getLogger(__name__)


def generate_plan(
    model_id: str,
    level: AdaptationLevel,
    params_b: float,
    hw: HardwareProfile,
) -> MigrationPlan:
    steps = _build_steps(model_id, level, params_b, hw)
    quant = _suggest_quant(params_b, hw)
    warnings = []

    if level in (AdaptationLevel.L3, AdaptationLevel.L4):
        warnings.append("Manual adaptation may be required — verify after conversion")

    vram_est = _estimate_vram(params_b, quant.bits if quant else 4)
    speed_est = _estimate_speed(params_b, hw, quant.bits if quant else 4)

    return MigrationPlan(
        model_id=model_id,
        level=level,
        steps=steps,
        quantize_suggestion=quant,
        estimated_vram_gb=round(vram_est, 2),
        estimated_speed_tok_per_sec=round(speed_est, 1),
        warnings=warnings,
    )


def _build_steps(
    model_id: str,
    level: AdaptationLevel,
    params_b: float,
    hw: HardwareProfile,
) -> list[str]:
    steps = []

    if level == AdaptationLevel.L0:
        steps.append(f"Model '{model_id}' is natively supported — no conversion needed")
        steps.append("Load model directly via inference API")
        return steps

    if level in (AdaptationLevel.L1, AdaptationLevel.L2):
        steps.append("Download model weights from HuggingFace")
        steps.append("Convert weights to MLX format via Fusion-MLX")

    if level == AdaptationLevel.L1:
        steps.append("Standard components detected — auto-migration should succeed")

    if level == AdaptationLevel.L2:
        steps.append("Some ops may need code generation — verify conversion output")

    if level == AdaptationLevel.L3:
        steps.append("Download model weights from HuggingFace")
        steps.append("Manual adaptation required — identify missing ops")
        steps.append("Implement custom MLX ops for missing components")
        steps.append("Convert and test the adapted model")

    if level == AdaptationLevel.L4:
        steps.append("Model architecture is not supported on MLX")
        steps.append("Consider alternative models with similar capabilities")
        steps.append("Request community support for adaptation")
        return steps

    bits = 4 if params_b > 7 else 8
    if hw.effective_vram_gb < params_b * 2:
        bits = 4
    steps.append(f"Apply {bits}-bit quantization for optimal performance")

    steps.append("Register converted model in Hub")
    steps.append("Run inference benchmark to verify performance")

    return steps


def _suggest_quant(params_b: float, hw: HardwareProfile) -> QuantizeSuggestion:
    vram = hw.effective_vram_gb

    if vram <= 0:
        return QuantizeSuggestion(bits=4, reason="Unknown hardware — defaulting to Q4 for safety")

    fp16_vram = params_b * 2.0
    q8_vram = params_b * 1.0
    q4_vram = params_b * 0.5

    if fp16_vram <= vram * 0.7:
        return QuantizeSuggestion(
            bits=16,
            reason="Sufficient VRAM for FP16 — best quality",
            vram_estimate_gb=round(fp16_vram, 2),
        )
    if q8_vram <= vram * 0.7:
        return QuantizeSuggestion(
            bits=8,
            reason="Good VRAM for Q8 — high quality",
            vram_estimate_gb=round(q8_vram, 2),
        )
    if q4_vram <= vram * 0.7:
        return QuantizeSuggestion(
            bits=4,
            reason="Q4 recommended — fits available VRAM",
            vram_estimate_gb=round(q4_vram, 2),
        )

    return QuantizeSuggestion(
        bits=4,
        reason="Q4 minimum — model may be tight on VRAM",
        vram_estimate_gb=round(q4_vram, 2),
    )


def _estimate_vram(params_b: float, bits: int) -> float:
    bytes_per_param = bits / 8.0
    return params_b * bytes_per_param


def _estimate_speed(params_b: float, hw: HardwareProfile, bits: int) -> float:
    if not hw.gpu or hw.gpu.memory_bandwidth_gbps <= 0:
        return 0.0

    bandwidth = hw.gpu.memory_bandwidth_gbps
    vram = _estimate_vram(params_b, bits)
    if vram <= 0:
        return 0.0

    bytes_total = vram * 1e9
    efficiency = 0.82  # Apple Silicon backend factor
    return (bandwidth * 1e9 * efficiency) / bytes_total
