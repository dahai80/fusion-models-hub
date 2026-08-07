from __future__ import annotations

import logging
from typing import Any

import httpx

from ..hardware.detector import HardwareDetector
from ..hardware.types import HardwareProfile
from .scorer import build_reason, score_hardware_fit, score_popularity, score_quality, score_speed, weighted_total
from .types import ModelRecommendation, RecommendRequest, RecommendResponse

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50


class RecommendEngine:
    def __init__(self, mlx_url: str = "http://localhost:11432"):
        self.mlx_url = mlx_url.rstrip("/")
        self._hw_detector = HardwareDetector(mlx_url)

    async def recommend(
        self,
        request: RecommendRequest,
        models_from_db: list[dict[str, Any]],
    ) -> RecommendResponse:
        hw = await self._hw_detector.detect()

        candidates = [
            m for m in models_from_db
            if request.min_params_b <= m.get("params_b", 0) <= request.max_params_b
        ]
        if request.task and request.task != "all":
            candidates = [m for m in candidates if m.get("task", "llm") == request.task]

        logger.info("Evaluating %d models for task=%s preference=%s", len(candidates), request.task, request.preference)

        mlx_results = await self._batch_recommend_mlx(candidates, request.task, request.preference)

        results: list[ModelRecommendation] = []
        for model in candidates:
            model_id = model.get("id", model.get("model_id", "unknown"))
            mlx_data = mlx_results.get(model_id)
            rec = self._build_recommendation(model, hw, request.preference, mlx_data)
            results.append(rec)

        results.sort(key=lambda r: r.rank_score, reverse=True)
        results = results[: request.max_results]

        hw_summary = {
            "chip": hw.gpu.chip_generation.value if hw.gpu else "Unknown",
            "vram_gb": hw.effective_vram_gb,
            "ram_gb": hw.ram_gb,
        }

        return RecommendResponse(
            recommendations=results,
            hardware_summary=hw_summary,
            total_evaluated=len(candidates),
        )

    async def _batch_recommend_mlx(
        self,
        candidates: list[dict[str, Any]],
        task: str,
        preference: str,
    ) -> dict[str, dict[str, Any]]:
        if not candidates:
            return {}

        specs = []
        for m in candidates:
            model_id = m.get("id", m.get("model_id", "unknown"))
            params_b = m.get("params_b", 0)
            quant_type = m.get("quant_type", "Q4_K_M")
            specs.append({
                "model_id": model_id,
                "params": max(1, int(params_b * 1e9)),
                "quant_type": quant_type,
            })

        all_results: dict[str, dict[str, Any]] = {}
        for i in range(0, len(specs), _BATCH_SIZE):
            batch = specs[i : i + _BATCH_SIZE]
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{self.mlx_url}/v1/recommend/batch",
                        json={
                            "models": batch,
                            "task": task or None,
                            "preference": preference,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for r in data.get("results", []):
                            mid = r.get("model_id", "")
                            all_results[mid] = r
                        logger.info(
                            "MLX batch recommend returned %d results (batch %d/%d)",
                            len(data.get("results", [])),
                            i // _BATCH_SIZE + 1,
                            (len(specs) + _BATCH_SIZE - 1) // _BATCH_SIZE,
                        )
                    else:
                        logger.warning("MLX batch recommend returned %d: %s", resp.status_code, resp.text)
                        for s in batch:
                            all_results[s["model_id"]] = await self._single_recommend_mlx(s)
            except Exception as e:
                logger.warning("MLX batch recommend failed: %s, falling back to single", e)
                for s in batch:
                    all_results[s["model_id"]] = await self._single_recommend_mlx(s)

        return all_results

    async def _single_recommend_mlx(self, spec: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.mlx_url}/v1/recommend",
                    json={
                        "model_id": spec["model_id"],
                        "params": spec["params"],
                        "quant_type": spec.get("quant_type", "Q4_K_M"),
                    },
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            logger.debug("MLX single recommend failed for %s: %s", spec.get("model_id"), e)
        return {}

    @staticmethod
    def _build_recommendation(
        model: dict[str, Any],
        hw: HardwareProfile,
        preference: str,
        mlx_data: dict[str, Any] | None,
    ) -> ModelRecommendation:
        model_id = model.get("id", model.get("model_id", "unknown"))
        params_b = model.get("params_b", 0)
        quant_type = model.get("quant_type", "Q4_K_M")

        can_run = False
        fit_type = "none"
        vram_required_gb = 0.0
        vram_available_gb = hw.effective_vram_gb
        tok_per_sec = 0.0

        if mlx_data:
            can_run = mlx_data.get("can_run", False)
            fit_type = mlx_data.get("fit_type", "none")
            vram_required_gb = mlx_data.get("vram_required_gb", 0.0)
            vram_available_gb = mlx_data.get("vram_available_gb", vram_available_gb)
            tok_per_sec = mlx_data.get("estimated_tok_per_sec", 0.0)

        hw_score = score_hardware_fit(vram_required_gb, vram_available_gb, can_run)
        q_score = score_quality(quant_type)
        s_score = score_speed(tok_per_sec)
        p_score = score_popularity(model.get("download_count", 0))
        rank = weighted_total(hw_score, q_score, s_score, p_score, preference)

        if not can_run:
            rank = 0.0

        return ModelRecommendation(
            model_id=model_id,
            name=model.get("name", model_id),
            task=model.get("task", "llm"),
            params_b=params_b,
            quant_type=quant_type,
            can_run=can_run,
            fit_type=fit_type,
            vram_required_gb=round(vram_required_gb, 2),
            vram_available_gb=round(vram_available_gb, 2),
            estimated_tok_per_sec=round(tok_per_sec, 1),
            rank_score=round(rank, 1),
            quality_score=round(q_score, 1),
            speed_score=round(s_score, 1),
            hardware_score=round(hw_score, 1),
            popularity_score=round(p_score, 1),
            reason=build_reason(can_run, hw_score, q_score, s_score, preference),
        )
