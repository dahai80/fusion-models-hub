import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ModelType(str, enum.Enum):
    LLM = "llm"
    CHAT = "chat"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    LORA = "lora"
    CODE = "code"
    AUDIO = "audio"
    IMAGE = "image"


class ModelFormat(str, enum.Enum):
    MLX = "mlx"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PYTORCH = "pytorch"
    ONNX = "onnx"


class Quantization(str, enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    Q8 = "8bit"
    Q6 = "6bit"
    Q4 = "4bit"
    Q2 = "2bit"
    NONE = "none"


class VersionStatus(str, enum.Enum):
    DRAFT = "draft"
    TESTING = "testing"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class ModelSource(str, enum.Enum):
    OFFICIAL = "official"
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    COMMUNITY = "community"
    CONVERTED = "converted"
    LOCAL = "local"


class ModelStatus(str, enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid4() -> str:
    return uuid.uuid4().hex[:16]


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    tenant_id: Mapped[str] = mapped_column(String(16), ForeignKey("tenants.id"), default="")
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    model_type: Mapped[ModelType] = mapped_column(Enum(ModelType), default=ModelType.LLM)
    architecture: Mapped[str] = mapped_column(String(64), default="")
    params_size: Mapped[str] = mapped_column(String(16), default="")
    license: Mapped[str] = mapped_column(String(64), default="")
    author: Mapped[str] = mapped_column(String(128), default="")
    language: Mapped[str] = mapped_column(String(128), default="")
    task_types: Mapped[str] = mapped_column(String(256), default="")
    owner: Mapped[str] = mapped_column(String(128), default="")
    hf_repo: Mapped[str] = mapped_column(String(256), default="")
    download_count: Mapped[int] = mapped_column(Integer, default=0)
    model_modules: Mapped[str] = mapped_column(String(256), default="")
    idle_timeout_minutes: Mapped[int] = mapped_column(Integer, default=60)
    ttl_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    model_status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.DRAFT)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", lazy="selectin",
    )
    tags: Mapped[list["ModelTag"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", lazy="selectin",
    )
    ratings: Mapped[list["ModelRating"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", lazy="selectin",
    )
    favorites: Mapped[list["ModelFavorite"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", lazy="selectin",
    )
    branches: Mapped[list["ModelBranch"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", lazy="selectin",
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    format: Mapped[ModelFormat] = mapped_column(Enum(ModelFormat), default=ModelFormat.MLX)
    quantization: Mapped[Quantization] = mapped_column(Enum(Quantization), default=Quantization.Q4)
    status: Mapped[VersionStatus] = mapped_column(Enum(VersionStatus), default=VersionStatus.DRAFT)
    file_path: Mapped[str] = mapped_column(String(512), default="")
    file_hash: Mapped[str] = mapped_column(String(64), default="")
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    release_notes: Mapped[str] = mapped_column(Text, default="")
    benchmark_score: Mapped[float] = mapped_column(Float, default=0.0)
    inference_latency: Mapped[float] = mapped_column(Float, default=0.0)
    throughput: Mapped[float] = mapped_column(Float, default=0.0)
    memory_usage: Mapped[float] = mapped_column(Float, default=0.0)
    context_length: Mapped[int] = mapped_column(Integer, default=0)
    successor_version_id: Mapped[str] = mapped_column(String(16), default="")
    encrypted: Mapped[bool] = mapped_column(Boolean, default=False)
    license_type: Mapped[str] = mapped_column(String(64), default="")
    data_compliance: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    model: Mapped["Model"] = relationship(back_populates="versions")


class ModelTag(Base):
    __tablename__ = "model_tags"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(String(128), default="")

    model: Mapped["Model"] = relationship(back_populates="tags")


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QuantizeTask(Base):
    __tablename__ = "quantize_tasks"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    source_version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), nullable=False)
    target_format: Mapped[str] = mapped_column(String(32), default="mlx")
    quant_bits: Mapped[int] = mapped_column(Integer, default=4)
    calibration_dataset: Mapped[str] = mapped_column(String(256), default="")
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.PENDING)
    output_version_id: Mapped[str] = mapped_column(String(16), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    tenant_id: Mapped[str] = mapped_column(String(16), ForeignKey("tenants.id"), default="")
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(8), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    permissions: Mapped[str] = mapped_column(String(128), default="read,write")
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.DEVELOPER)
    qps_limit: Mapped[int] = mapped_column(Integer, default=0)
    allowed_models: Mapped[str] = mapped_column(String(512), default="")
    allowed_modules: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    tenant_id: Mapped[str] = mapped_column(String(16), ForeignKey("tenants.id"), default="")
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(16), default="")
    api_key_id: Mapped[str] = mapped_column(String(16), default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ClusterNode(Base):
    __tablename__ = "cluster_nodes"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    url: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="active")
    capabilities: Mapped[str] = mapped_column(String(256), default="inference,quantize")
    last_heartbeat: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class WebhookEvent(str, enum.Enum):
    MODEL_CREATED = "model.created"
    MODEL_DELETED = "model.deleted"
    VERSION_PUBLISHED = "version.published"
    VERSION_DEPRECATED = "version.deprecated"
    QUANTIZE_COMPLETED = "quantize.completed"
    QUANTIZE_FAILED = "quantize.failed"


class Webhook(Base):
    __tablename__ = "webhooks"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    tenant_id: Mapped[str] = mapped_column(String(16), ForeignKey("tenants.id"), default="")
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    secret: Mapped[str] = mapped_column(String(128), default="")
    events: Mapped[str] = mapped_column(String(512), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class DeploymentStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    tenant_id: Mapped[str] = mapped_column(String(16), ForeignKey("tenants.id"), default="")
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    replicas: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[DeploymentStatus] = mapped_column(Enum(DeploymentStatus), default=DeploymentStatus.PENDING)
    gray_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    gray_version_id: Mapped[str] = mapped_column(String(16), default="")
    gray_traffic_ratio: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class EvaluationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    tenant_id: Mapped[str] = mapped_column(String(16), ForeignKey("tenants.id"), default="")
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    benchmark_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[EvaluationStatus] = mapped_column(Enum(EvaluationStatus), default=EvaluationStatus.PENDING)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    metrics: Mapped[str] = mapped_column(Text, default="{}")
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ScanStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SecurityScan(Base):
    __tablename__ = "security_scans"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    scan_type: Mapped[str] = mapped_column(String(32), default="full")
    status: Mapped[ScanStatus] = mapped_column(Enum(ScanStatus), default=ScanStatus.PENDING)
    findings: Mapped[str] = mapped_column(Text, default="{}")
    risk_level: Mapped[str] = mapped_column(String(16), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Watermark(Base):
    __tablename__ = "watermarks"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    watermark_type: Mapped[str] = mapped_column(String(32), default="metadata")
    payload: Mapped[str] = mapped_column(Text, default="{}")
    signature: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ApprovalLevel(str, enum.Enum):
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    tenant_id: Mapped[str] = mapped_column(String(16), ForeignKey("tenants.id"), default="")
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    level: Mapped[ApprovalLevel] = mapped_column(Enum(ApprovalLevel), default=ApprovalLevel.L1)
    status: Mapped[ApprovalStatus] = mapped_column(Enum(ApprovalStatus), default=ApprovalStatus.PENDING)
    requester: Mapped[str] = mapped_column(String(128), default="")
    approver: Mapped[str] = mapped_column(String(128), default="")
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class LoraMergeTask(Base):
    __tablename__ = "lora_merge_tasks"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    base_version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), nullable=False)
    lora_version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), nullable=False)
    target_format: Mapped[str] = mapped_column(String(32), default="mlx")
    quant_bits: Mapped[int] = mapped_column(Integer, default=4)
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.PENDING)
    output_version_id: Mapped[str] = mapped_column(String(16), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DistributedTaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DistributedTask(Base):
    __tablename__ = "distributed_tasks"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    target_nodes: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[DistributedTaskStatus] = mapped_column(
        Enum(DistributedTaskStatus), default=DistributedTaskStatus.PENDING,
    )
    progress: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GitLfsLock(Base):
    __tablename__ = "gitlfs_locks"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    owner: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ModelRating(Base):
    __tablename__ = "model_ratings"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), default="")
    score: Mapped[int] = mapped_column(Integer, default=0)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    model: Mapped["Model"] = relationship(back_populates="ratings")


class ModelFavorite(Base):
    __tablename__ = "model_favorites"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    model: Mapped["Model"] = relationship(back_populates="favorites")


class BranchStatus(str, enum.Enum):
    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"


class ModelBranch(Base):
    __tablename__ = "model_branches"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    model_id: Mapped[str] = mapped_column(String(16), ForeignKey("models.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    base_version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    head_version_id: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.id"), default="")
    status: Mapped[BranchStatus] = mapped_column(Enum(BranchStatus), default=BranchStatus.ACTIVE)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    model: Mapped["Model"] = relationship(back_populates="branches")


class DownloadTask(Base):
    __tablename__ = "download_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("models.id"), nullable=False)
    version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)
    downloaded_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    total_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    speed_limit_kbps: Mapped[int] = mapped_column(Integer, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    error_message: Mapped[str] = mapped_column(Text, default="")
    file_path: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
