from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class AdaptationLevel(str, enum.Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class MigrationCost(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    extreme = "extreme"


class CompileStrategy(str, enum.Enum):
    block_loop = "block+loop"
    full = "full"


class AdaptationResult(BaseModel):
    model_id: str
    level: AdaptationLevel
    level_desc: str
    migration_cost: MigrationCost
    components_matched: list[str] = Field(default_factory=list)
    missing_ops: list[str] = Field(default_factory=list)
    compile_strategy: CompileStrategy = CompileStrategy.block_loop
    warnings: list[str] = Field(default_factory=list)


class QuantizeSuggestion(BaseModel):
    bits: int
    reason: str
    vram_estimate_gb: float = 0.0
    speed_estimate_tok_per_sec: float = 0.0


class MigrationPlan(BaseModel):
    model_id: str
    level: AdaptationLevel
    steps: list[str] = Field(default_factory=list)
    quantize_suggestion: QuantizeSuggestion | None = None
    estimated_vram_gb: float = 0.0
    estimated_speed_tok_per_sec: float = 0.0
    warnings: list[str] = Field(default_factory=list)
