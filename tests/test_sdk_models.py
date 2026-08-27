import logging

import pytest
from pydantic import ValidationError

from fusion_model_hub.sdk.models import (
    ApprovalActionRequest,
    ApprovalCreateRequest,
    BranchCreateRequest,
    BranchUpdateRequest,
    EncryptionRequest,
    HealthResponse,
    LoraMergeRequest,
    ModelCreate,
    ModelResponse,
    ModelUpdate,
    PaginatedResponse,
    QuantizeRequest,
    RatingCreateRequest,
    SecurityScanRequest,
    StatusChangeRequest,
    TaskResponse,
    VersionCreate,
    VersionResponse,
    VersionUpdate,
    WatermarkEmbedRequest,
)

logger = logging.getLogger(__name__)


class TestModelCreate:
    def test_minimal(self):
        m = ModelCreate(name="test-model")
        assert m.name == "test-model"
        assert m.description == ""
        assert m.model_type == "llm"

    def test_full_fields(self):
        m = ModelCreate(
            name="test",
            description="desc",
            model_type="embedding",
            architecture="transformer",
            params_size="7B",
            license="mit",
            author="acme",
            language="en",
            task_types="chat",
            owner="team-a",
            hf_repo="org/repo",
        )
        assert m.params_size == "7B"
        assert m.hf_repo == "org/repo"

    def test_name_required(self):
        with pytest.raises(ValidationError):
            ModelCreate()

    def test_name_empty_fails(self):
        with pytest.raises(ValidationError):
            ModelCreate(name="")

    def test_name_too_long(self):
        with pytest.raises(ValidationError):
            ModelCreate(name="x" * 65)


class TestModelUpdate:
    def test_all_none(self):
        m = ModelUpdate()
        assert m.description is None
        assert m.model_type is None

    def test_partial(self):
        m = ModelUpdate(description="updated")
        assert m.description == "updated"
        assert m.model_type is None


class TestModelResponse:
    def test_required_fields(self):
        r = ModelResponse(id="m1", name="test")
        assert r.id == "m1"
        assert r.download_count == 0
        assert r.created_at is None

    def test_full(self):
        r = ModelResponse(id="m1", name="test", download_count=42, created_at="2025-01-01")
        assert r.download_count == 42
        assert r.created_at == "2025-01-01"


class TestVersionCreate:
    def test_required(self):
        v = VersionCreate(version="1.0.0")
        assert v.version == "1.0.0"
        assert v.format == "mlx"

    def test_full(self):
        v = VersionCreate(version="2.0.0", format="gguf", quantization="8bit", release_notes="major")
        assert v.format == "gguf"
        assert v.quantization == "8bit"

    def test_version_required(self):
        with pytest.raises(ValidationError):
            VersionCreate()


class TestVersionUpdate:
    def test_all_none(self):
        v = VersionUpdate()
        assert v.file_path is None
        assert v.benchmark_score is None

    def test_partial(self):
        v = VersionUpdate(benchmark_score=95.5, encrypted=True)
        assert v.benchmark_score == 95.5
        assert v.encrypted is True


class TestVersionResponse:
    def test_required(self):
        r = VersionResponse(id="v1", model_id="m1", version="1.0.0")
        assert r.id == "v1"
        assert r.status == "draft"
        assert r.encrypted is False

    def test_full(self):
        r = VersionResponse(
            id="v1",
            model_id="m1",
            version="1.0.0",
            status="published",
            benchmark_score=88.0,
            encrypted=True,
        )
        assert r.status == "published"
        assert r.encrypted is True


class TestStatusChangeRequest:
    def test_required(self):
        r = StatusChangeRequest(target_status="published")
        assert r.target_status == "published"
        assert r.remark == ""
        assert r.approval_level == "l1"

    def test_full(self):
        r = StatusChangeRequest(target_status="deprecated", remark="old", approval_level="l2")
        assert r.approval_level == "l2"

    def test_target_status_required(self):
        with pytest.raises(ValidationError):
            StatusChangeRequest()


class TestQuantizeRequest:
    def test_required(self):
        r = QuantizeRequest(source_version_id="v1")
        assert r.source_version_id == "v1"
        assert r.target_format == "mlx"
        assert r.quant_bits == 4

    def test_full(self):
        r = QuantizeRequest(
            source_version_id="v1", target_format="gguf", quant_bits=8, calibration_dataset="data.jsonl"
        )
        assert r.quant_bits == 8
        assert r.calibration_dataset == "data.jsonl"


class TestLoraMergeRequest:
    def test_required(self):
        r = LoraMergeRequest(base_version_id="v1", lora_version_id="v2")
        assert r.base_version_id == "v1"
        assert r.lora_version_id == "v2"
        assert r.target_format == "mlx"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            LoraMergeRequest(base_version_id="v1")


class TestSecurityScanRequest:
    def test_required(self):
        r = SecurityScanRequest(version_id="v1")
        assert r.version_id == "v1"
        assert r.scan_type == "full"

    def test_custom_type(self):
        r = SecurityScanRequest(version_id="v1", scan_type="quick")
        assert r.scan_type == "quick"


class TestWatermarkEmbedRequest:
    def test_required(self):
        r = WatermarkEmbedRequest(version_id="v1")
        assert r.version_id == "v1"
        assert r.metadata == "{}"

    def test_custom_metadata(self):
        r = WatermarkEmbedRequest(version_id="v1", metadata='{"owner":"acme"}')
        assert r.metadata == '{"owner":"acme"}'


class TestEncryptionRequest:
    def test_required(self):
        r = EncryptionRequest(version_id="v1")
        assert r.version_id == "v1"

    def test_missing(self):
        with pytest.raises(ValidationError):
            EncryptionRequest()


class TestApprovalCreateRequest:
    def test_required(self):
        r = ApprovalCreateRequest(model_id="m1")
        assert r.model_id == "m1"
        assert r.level == "l1"
        assert r.version_id == ""

    def test_full(self):
        r = ApprovalCreateRequest(model_id="m1", version_id="v1", level="l2", requester="admin", reason="deploy")
        assert r.level == "l2"
        assert r.requester == "admin"


class TestApprovalActionRequest:
    def test_default(self):
        r = ApprovalActionRequest()
        assert r.comment == ""

    def test_with_comment(self):
        r = ApprovalActionRequest(comment="approved")
        assert r.comment == "approved"


class TestRatingCreateRequest:
    def test_valid(self):
        r = RatingCreateRequest(score=5)
        assert r.score == 5
        assert r.comment == ""

    def test_min_score(self):
        r = RatingCreateRequest(score=1)
        assert r.score == 1

    def test_score_below_min(self):
        with pytest.raises(ValidationError):
            RatingCreateRequest(score=0)

    def test_score_above_max(self):
        with pytest.raises(ValidationError):
            RatingCreateRequest(score=6)

    def test_with_comment(self):
        r = RatingCreateRequest(score=3, comment="average")
        assert r.comment == "average"


class TestBranchCreateRequest:
    def test_required(self):
        r = BranchCreateRequest(name="feature-x")
        assert r.name == "feature-x"
        assert r.base_version_id == ""
        assert r.description == ""

    def test_name_empty_fails(self):
        with pytest.raises(ValidationError):
            BranchCreateRequest(name="")

    def test_name_too_long(self):
        with pytest.raises(ValidationError):
            BranchCreateRequest(name="x" * 65)

    def test_full(self):
        r = BranchCreateRequest(name="feature-y", base_version_id="v1", description="test branch")
        assert r.base_version_id == "v1"


class TestBranchUpdateRequest:
    def test_all_none(self):
        r = BranchUpdateRequest()
        assert r.head_version_id is None
        assert r.status is None
        assert r.description is None

    def test_partial(self):
        r = BranchUpdateRequest(status="merged", description="done")
        assert r.status == "merged"


class TestPaginatedResponse:
    def test_required(self):
        r = PaginatedResponse(items=[], total=0, page=1, page_size=10)
        assert r.items == []
        assert r.total == 0

    def test_with_items(self):
        r = PaginatedResponse(items=[{"id": "1"}], total=1, page=1, page_size=10)
        assert len(r.items) == 1


class TestTaskResponse:
    def test_required(self):
        r = TaskResponse(id="t1", status="running")
        assert r.id == "t1"
        assert r.status == "running"
        assert r.error_message == ""
        assert r.created_at is None

    def test_full(self):
        r = TaskResponse(id="t1", status="failed", error_message="OOM", created_at="2025-01-01")
        assert r.error_message == "OOM"


class TestHealthResponse:
    def test_required(self):
        r = HealthResponse(status="healthy")
        assert r.status == "healthy"
        assert r.version == "0.1.0"

    def test_custom_version(self):
        r = HealthResponse(status="degraded", version="1.2.3")
        assert r.version == "1.2.3"
