from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    task: str = Field("llm", description="Task type: llm|text2image|text2video|embedding")
    preference: str = Field("balanced", description="Preference: quality|balanced|speed")
    max_results: int = Field(10, ge=1, le=50, description="Max results to return")
    min_params_b: float = Field(0, ge=0, description="Min model size in billions")
    max_params_b: float = Field(1000, ge=0, description="Max model size in billions")


class ModelRecommendation(BaseModel):
    model_id: str
    name: str
    task: str
    params_b: float
    quant_type: str
    can_run: bool
    fit_type: str
    vram_required_gb: float
    vram_available_gb: float
    estimated_tok_per_sec: float
    rank_score: float
    quality_score: float
    speed_score: float
    hardware_score: float
    popularity_score: float
    reason: str


class RecommendResponse(BaseModel):
    recommendations: list[ModelRecommendation]
    hardware_summary: dict
    total_evaluated: int
