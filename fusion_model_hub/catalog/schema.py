import logging

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# Issue #102: static model catalog format. One declarative JSON file per
# model, versioned in-repo (catalog/models/*.json). Static files beat runtime
# discovery for cross-node consistency: lintable, diffable, consumable by
# every fusion-* project without querying a live hub.

SCHEMA_VERSION = 1
MODALITIES = {"text", "image", "audio", "video"}
CAPABILITIES = {"functionCall", "reasoning", "vision", "embedding"}


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = "USD"
    inputPerMillionTokens: float = Field(0.0, ge=0)
    outputPerMillionTokens: float = Field(0.0, ge=0)


class Modality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: list[str]
    output: list[str]

    @model_validator(mode="after")
    def _check_values(self):
        for side in ("input", "output"):
            values = getattr(self, side)
            if not values:
                raise ValueError(f"modality.{side} must list at least one modality")
            bad = set(values) - MODALITIES
            if bad:
                raise ValueError(f"modality.{side} unknown values: {sorted(bad)} (allowed: {sorted(MODALITIES)})")
        return self


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functionCall: bool = False
    reasoning: bool = False
    vision: bool = False
    embedding: bool = False


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schemaVersion: int
    name: str = Field(..., min_length=1)
    hubId: str = ""
    revision: str = ""
    contextWindowTokens: int = Field(..., gt=0)
    maxOutputTokens: int = Field(..., gt=0)
    pricing: Pricing = Field(default_factory=Pricing)
    modality: Modality
    capabilities: Capabilities = Field(default_factory=Capabilities)
    quantizations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        if self.schemaVersion != SCHEMA_VERSION:
            raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}, got {self.schemaVersion}")
        if self.maxOutputTokens > self.contextWindowTokens:
            raise ValueError(
                f"maxOutputTokens ({self.maxOutputTokens}) exceeds contextWindowTokens ({self.contextWindowTokens})"
            )
        return self


def validate_entry(data: dict, source: str = "") -> CatalogEntry:
    try:
        return CatalogEntry.model_validate(data)
    except Exception as e:
        raise ValueError(f"{source}: {e}") from e


def validate_dir(path) -> list[CatalogEntry]:
    # Returns validated entries; raises on the first invalid file so schema
    # drift fails loudly (CI gate + hub-side read guard).
    import json
    from pathlib import Path

    entries = []
    for f in sorted(Path(path).glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except ValueError as e:
            raise ValueError(f"{f}: invalid JSON: {e}") from e
        entries.append(validate_entry(data, source=str(f)))
    return entries
