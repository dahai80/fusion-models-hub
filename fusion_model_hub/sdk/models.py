from pydantic import BaseModel, Field


class ModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = ""
    model_type: str = "llm"
    architecture: str = ""
    params_size: str = ""
    license: str = ""
    author: str = ""
    language: str = ""
    task_types: str = ""
    owner: str = ""
    hf_repo: str = ""


class ModelUpdate(BaseModel):
    description: str | None = None
    model_type: str | None = None
    architecture: str | None = None
    params_size: str | None = None
    license: str | None = None
    author: str | None = None
    language: str | None = None
    task_types: str | None = None
    owner: str | None = None
    hf_repo: str | None = None


class ModelResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    model_type: str = "llm"
    architecture: str = ""
    params_size: str = ""
    license: str = ""
    author: str = ""
    language: str = ""
    task_types: str = ""
    owner: str = ""
    hf_repo: str = ""
    download_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class VersionCreate(BaseModel):
    version: str
    format: str = "mlx"
    quantization: str = "4bit"
    release_notes: str = ""


class VersionUpdate(BaseModel):
    file_path: str | None = None
    file_hash: str | None = None
    file_size: int | None = None
    release_notes: str | None = None
    benchmark_score: float | None = None
    inference_latency: float | None = None
    throughput: float | None = None
    memory_usage: float | None = None
    context_length: int | None = None
    successor_version_id: str | None = None
    encrypted: bool | None = None
    license_type: str | None = None
    data_compliance: str | None = None


class VersionResponse(BaseModel):
    id: str
    model_id: str
    version: str
    format: str = "mlx"
    quantization: str = "4bit"
    status: str = "draft"
    file_path: str = ""
    file_hash: str = ""
    file_size: int = 0
    release_notes: str = ""
    benchmark_score: float = 0.0
    inference_latency: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    context_length: int = 0
    successor_version_id: str = ""
    encrypted: bool = False
    license_type: str = ""
    data_compliance: str = "{}"
    created_at: str | None = None


class StatusChangeRequest(BaseModel):
    target_status: str
    remark: str = ""
    approval_level: str = "l1"


class QuantizeRequest(BaseModel):
    source_version_id: str
    target_format: str = "mlx"
    quant_bits: int = 4
    calibration_dataset: str = ""


class LoraMergeRequest(BaseModel):
    base_version_id: str
    lora_version_id: str
    target_format: str = "mlx"
    quant_bits: int = 4


class SecurityScanRequest(BaseModel):
    version_id: str
    scan_type: str = "full"


class WatermarkEmbedRequest(BaseModel):
    version_id: str
    metadata: str = "{}"


class EncryptionRequest(BaseModel):
    version_id: str


class ApprovalCreateRequest(BaseModel):
    model_id: str
    version_id: str = ""
    level: str = "l1"
    requester: str = ""
    reason: str = ""


class ApprovalActionRequest(BaseModel):
    comment: str = ""


class RatingCreateRequest(BaseModel):
    score: int = Field(..., ge=1, le=5)
    comment: str = ""


class BranchCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    base_version_id: str = ""
    description: str = ""


class BranchUpdateRequest(BaseModel):
    head_version_id: str | None = None
    status: str | None = None
    description: str | None = None


class PaginatedResponse(BaseModel):
    items: list[dict]
    total: int
    page: int
    page_size: int


class TaskResponse(BaseModel):
    id: str
    status: str
    error_message: str = ""
    created_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
