from __future__ import annotations

import logging

import httpx

from ..hardware.detector import HardwareDetector
from .migration import generate_plan
from .types import AdaptationLevel, AdaptationResult, CompileStrategy, MigrationCost, MigrationPlan

logger = logging.getLogger(__name__)

_LEVEL_DESCRIPTIONS = {
    AdaptationLevel.L0: "Natively supported — no conversion needed",
    AdaptationLevel.L1: "Component-level auto-migration — standard components detected",
    AdaptationLevel.L2: "Minor code generation needed — some ops require custom MLX implementation",
    AdaptationLevel.L3: "Manual adaptation required — critical ops missing",
    AdaptationLevel.L4: "Not supported — architecture incompatible",
}

_COST_MAP = {
    AdaptationLevel.L0: MigrationCost.low,
    AdaptationLevel.L1: MigrationCost.low,
    AdaptationLevel.L2: MigrationCost.medium,
    AdaptationLevel.L3: MigrationCost.high,
    AdaptationLevel.L4: MigrationCost.extreme,
}


class AdaptDecisionEngine:
    def __init__(self, mlx_url: str = "http://localhost:11432"):
        self.mlx_url = mlx_url.rstrip("/")
        self._hw_detector = HardwareDetector(mlx_url)

    async def assess(
        self,
        model_id: str,
        hf_repo: str | None = None,
        source_format: str | None = None,
    ) -> AdaptationResult:
        result = await self._call_mlx_migration_level(model_id, hf_repo, source_format)
        if result:
            if result.level in (AdaptationLevel.L2, AdaptationLevel.L3):
                analysis = await self._call_mlx_analyze(model_id, hf_repo)
                if analysis:
                    layer_types = analysis.get("layer_types", [])
                    result.components_matched = layer_types
                    special_ops = analysis.get("special_ops", [])
                    if special_ops:
                        result.missing_ops = list(set(result.missing_ops + special_ops))
                    params_info = analysis.get("params_by_layer", {})
                    if params_info:
                        result.warnings.append(
                            f"Parameter distribution: attention={params_info.get('attention', 0)}, "
                            f"ffn={params_info.get('ffn', 0)}, embedding={params_info.get('embedding', 0)}"
                        )
            return result

        logger.info("MLX migration-level API unavailable, using local fallback for %s", model_id)
        return self._local_fallback(model_id, hf_repo)

    async def assess_and_plan(
        self,
        model_id: str,
        params_b: float,
        hf_repo: str | None = None,
        source_format: str | None = None,
    ) -> MigrationPlan:
        adapt_result = await self.assess(model_id, hf_repo, source_format)
        hw = await self._hw_detector.detect()
        return generate_plan(model_id, adapt_result.level, params_b, hw)

    async def _call_mlx_migration_level(
        self, model_id: str, hf_repo: str | None, source_format: str | None,
    ) -> AdaptationResult | None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                body: dict = {"model_id": model_id}
                if hf_repo:
                    body["hf_repo"] = hf_repo
                if source_format:
                    body["source_format"] = source_format

                resp = await client.post(f"{self.mlx_url}/v1/migration-level", json=body)
                if resp.status_code != 200:
                    logger.debug("MLX migration-level returned %d", resp.status_code)
                    return None

                data = resp.json()
                return AdaptationResult(
                    model_id=model_id,
                    level=AdaptationLevel(data["level"]),
                    level_desc=data.get("level_desc", ""),
                    migration_cost=MigrationCost(data.get("migration_cost", "medium")),
                    components_matched=data.get("components_matched", []),
                    missing_ops=data.get("missing_ops", []),
                    compile_strategy=CompileStrategy(data.get("compile_strategy", "block+loop")),
                    warnings=data.get("warnings", []),
                )
        except Exception as e:
            logger.debug("MLX migration-level call failed: %s", e)
            return None

    async def _call_mlx_analyze(
        self, model_id: str, hf_repo: str | None,
    ) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                payload: dict = {}
                if hf_repo:
                    payload["hf_repo"] = hf_repo
                else:
                    payload["model_path"] = model_id

                resp = await client.post(f"{self.mlx_url}/v1/analyze", json=payload)
                if resp.status_code == 200:
                    logger.info("MLX analyze succeeded for %s", model_id)
                    return resp.json()
                logger.debug("MLX analyze returned %d for %s", resp.status_code, model_id)
                return None
        except Exception as e:
            logger.debug("MLX analyze call failed for %s: %s", model_id, e)
            return None

    @staticmethod
    def _local_fallback(model_id: str, hf_repo: str | None) -> AdaptationResult:
        known_prefixes = [
            "qwen", "llama", "mistral", "gemma", "phi", "deepseek", "yi",
            "chatglm", "solar", "internlm", "starcoder", "codellama",
        ]
        model_lower = model_id.lower()

        for prefix in known_prefixes:
            if prefix in model_lower:
                return AdaptationResult(
                    model_id=model_id,
                    level=AdaptationLevel.L0,
                    level_desc=_LEVEL_DESCRIPTIONS[AdaptationLevel.L0],
                    migration_cost=MigrationCost.low,
                    components_matched=["Transformer", "MHA/GQA", "FFN", "RoPE", "Norm"],
                    compile_strategy=CompileStrategy.block_loop,
                )

        return AdaptationResult(
            model_id=model_id,
            level=AdaptationLevel.L2,
            level_desc=_LEVEL_DESCRIPTIONS[AdaptationLevel.L2],
            migration_cost=MigrationCost.medium,
            missing_ops=["unknown structure — requires analysis"],
            compile_strategy=CompileStrategy.block_loop,
            warnings=["Local fallback assessment — use MLX /v1/migration-level for accurate results"],
        )
