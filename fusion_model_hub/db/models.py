import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
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
    COMMUNITY = "community"
    CONVERTED = "converted"
    LOCAL = "local"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid4() -> str:
    return uuid.uuid4().hex[:16]


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model", cascade="all, delete-orphan", lazy="selectin",
    )
    tags: Mapped[list["ModelTag"]] = relationship(
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
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.PENDING)
    output_version_id: Mapped[str] = mapped_column(String(16), default="")
    error_message: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(8), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    permissions: Mapped[str] = mapped_column(String(128), default="read,write")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=_uuid4)
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
