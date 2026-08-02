import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

from fusion_model_hub.sdk import FusionModelHubClient

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:11444"


def _mock_response(json_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


def _setup_mock_client(MockClient, method, response):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)
    getattr(mock_instance, method).return_value = response
    MockClient.return_value = mock_instance
    return mock_instance


@pytest.fixture
def sdk():
    return FusionModelHubClient(base_url=BASE_URL, api_key="test-key")


class TestClientInit:
    def test_base_url_strips_trailing_slash(self):
        c = FusionModelHubClient(base_url="http://host:9999/")
        assert c._base_url == "http://host:9999"

    def test_api_key_header(self):
        c = FusionModelHubClient(api_key="mykey")
        assert c._headers["X-API-Key"] == "mykey"

    def test_no_api_key_header(self):
        c = FusionModelHubClient()
        assert "X-API-Key" not in c._headers

    def test_url_construction(self):
        c = FusionModelHubClient(base_url="http://host:11444")
        assert c._url("/models") == "http://host:11444/api/v1/models"


class TestModelCRUD:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_create_model(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "m1", "name": "test-model"}
        _setup_mock_client(MockClient, "post", _mock_response(expected, 201))
        result = sdk.create_model({"name": "test-model"})
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_models(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_models()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_models_with_params(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [{"id": "m1"}], "total": 1}
        mock_c = _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_models(keyword="test")
        assert result == expected
        call_args = mock_c.get.call_args
        assert call_args[1]["params"] == {"keyword": "test"}

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_model(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "m1", "name": "test-model"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_model("m1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_update_model(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "m1", "name": "updated"}
        _setup_mock_client(MockClient, "put", _mock_response(expected))
        result = sdk.update_model("m1", {"name": "updated"})
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_delete_model(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"ok": True}
        _setup_mock_client(MockClient, "delete", _mock_response(expected))
        result = sdk.delete_model("m1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_import_from_hf(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "m2", "name": "hf-model"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.import_from_hf({"repo_id": "org/model"})
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_sync_registry(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"synced": 5}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.sync_registry("http://other-host:11444")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_batch_delete(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"deleted": 2}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.batch_delete(["m1", "m2"])
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_compare_models(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"models": []}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.compare_models(["m1", "m2"])
        assert result == expected


class TestVersionCRUD:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_versions(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_versions("m1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_versions_with_params(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        mock_c = _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_versions("m1", status="published")
        assert result == expected
        call_args = mock_c.get.call_args
        assert call_args[1]["params"] == {"status": "published"}

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "v1", "version": "1.0.0"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_version("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_update_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "v1", "status": "testing"}
        _setup_mock_client(MockClient, "put", _mock_response(expected))
        result = sdk.update_version("v1", {"status": "testing"})
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_delete_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"ok": True}
        _setup_mock_client(MockClient, "delete", _mock_response(expected))
        result = sdk.delete_version("v1")
        assert result == expected


class TestVersionActions:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_promote_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "v1", "status": "published"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.promote_version("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_benchmark_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "v1", "benchmark_score": 85.0}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.benchmark_version("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_rollback_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "v1", "status": "published"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.rollback_version("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_deprecate_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "v1", "status": "deprecated"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.deprecate_version("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_retire_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "v1", "status": "retired"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.retire_version("v1")
        assert result == expected


class TestQuantize:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_start_quantize(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "q1", "status": "pending"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.start_quantize("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_start_quantize_custom_params(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "q2", "status": "pending"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.start_quantize("v1", target_format="gguf", quant_bits=8)
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["target_format"] == "gguf"
        assert body["quant_bits"] == 8

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_quantize_tasks(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_quantize_tasks()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_quantize_status(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "q1", "status": "completed"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_quantize_status("q1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_start_lora_merge(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "lm1", "status": "pending"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.start_lora_merge("bv1", "lv1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_lora_merge_status(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "lm1", "status": "running"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_lora_merge_status("lm1")
        assert result == expected


class TestInference:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_chat_completions(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.chat_completions("test-model", [{"role": "user", "content": "hi"}])
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_chat_completions_with_kwargs(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"choices": []}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.chat_completions("test-model", [{"role": "user", "content": "hi"}], temperature=0.7)
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["temperature"] == 0.7

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_completions(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"choices": [{"text": "world"}]}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.completions("test-model", "hello")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_completions_with_kwargs(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"choices": []}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.completions("test-model", "hello", max_tokens=100)
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["max_tokens"] == 100

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_embeddings(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"data": [{"embedding": [0.1, 0.2]}]}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.embeddings("test-model", "hello world")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_embeddings_with_list_input(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"data": []}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.embeddings("test-model", ["hello", "world"])
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["input"] == ["hello", "world"]


class TestSecurity:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_start_security_scan(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "s1", "status": "pending"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.start_security_scan("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_start_security_scan_custom_type(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "s2", "status": "pending"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.start_security_scan("v1", scan_type="quick")
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["scan_type"] == "quick"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_security_scan(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "s1", "status": "completed"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_security_scan("s1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_security_scans(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_security_scans()
        assert result == expected


class TestWatermark:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_embed_watermark(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "w1", "status": "embedded"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.embed_watermark("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_embed_watermark_with_metadata(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "w2", "status": "embedded"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.embed_watermark("v1", metadata='{"owner":"test"}')
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["metadata"] == '{"owner":"test"}'

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_verify_watermark(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"verified": True}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.verify_watermark("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_watermarks(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_watermarks()
        assert result == expected


class TestEncryption:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_encrypt_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"status": "encrypting"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.encrypt_version("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_decrypt_version(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"status": "decrypting"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.decrypt_version("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_encryption_status(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"version_id": "v1", "encrypted": True}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_encryption_status("v1")
        assert result == expected


class TestApprovals:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_create_approval(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "a1", "status": "pending"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.create_approval("v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_create_approval_custom_params(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "a2", "status": "approved"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.create_approval("v1", level="L1", reason="auto")
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["level"] == "L1"
        assert body["reason"] == "auto"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_approvals(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_approvals()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_approval(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "a1", "status": "pending"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_approval("a1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_approve_request(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "a1", "status": "approved"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.approve_request("a1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_approve_request_with_comment(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "a1", "status": "approved"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.approve_request("a1", comment="looks good")
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["comment"] == "looks good"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_reject_request(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "a1", "status": "rejected"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.reject_request("a1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_reject_request_with_comment(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "a1", "status": "rejected"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.reject_request("a1", comment="not ready")
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["comment"] == "not ready"


class TestGitLFS:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_gitlfs_batch(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"objects": [{"oid": "abc123"}]}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.gitlfs_batch("upload", [{"oid": "abc123", "size": 1024}])
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_gitlfs_batch_download(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"objects": []}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.gitlfs_batch("download", [{"oid": "xyz", "size": 2048}])
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["operation"] == "download"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_create_gitlfs_lock(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"lock": {"id": "l1", "path": "models/w.bin"}}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.create_gitlfs_lock("models/w.bin")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_gitlfs_locks(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"locks": []}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_gitlfs_locks()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_delete_gitlfs_lock(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"ok": True}
        _setup_mock_client(MockClient, "delete", _mock_response(expected))
        result = sdk.delete_gitlfs_lock("l1")
        assert result == expected


class TestCluster:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_nodes(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_nodes()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_add_node(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "n1", "name": "node1"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.add_node("node1", "http://node1:11444")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_add_node_custom_capabilities(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "n2", "name": "node2"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.add_node("node2", "http://node2:11444", capabilities="inference")
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["capabilities"] == "inference"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_node(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "n1", "name": "node1"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_node("n1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_remove_node(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"ok": True}
        _setup_mock_client(MockClient, "delete", _mock_response(expected))
        result = sdk.remove_node("n1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_submit_distributed_task(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "dt1", "status": "pending"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.submit_distributed_task("benchmark", "v1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_submit_distributed_task_with_nodes(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "dt2", "status": "pending"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.submit_distributed_task("benchmark", "v1", target_node_ids=["n1", "n2"])
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["target_node_ids"] == ["n1", "n2"]

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_submit_distributed_task_with_config(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "dt3", "status": "pending"}
        mock_c = _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.submit_distributed_task("benchmark", "v1", config='{"key":"val"}')
        assert result == expected
        call_args = mock_c.post.call_args
        body = call_args[1]["json"]
        assert body["config"] == '{"key":"val"}'

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_distributed_task(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"task_id": "dt1", "status": "completed"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.get_distributed_task("dt1")
        assert result == expected


class TestSystem:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_health(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"status": "healthy"}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.health()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_storage_stats(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"total_bytes": 1024, "used_bytes": 512}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.storage_stats()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_export_data(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"models": []}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.export_data()
        assert result == expected


class TestAuth:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_create_api_key(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "k1", "name": "my-key", "key": "fmh-xxx"}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.create_api_key("my-key")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_list_api_keys(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"items": [], "total": 0}
        _setup_mock_client(MockClient, "get", _mock_response(expected))
        result = sdk.list_api_keys()
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_deactivate_api_key(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"id": "k1", "active": False}
        _setup_mock_client(MockClient, "post", _mock_response(expected))
        result = sdk.deactivate_api_key("k1")
        assert result == expected

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_delete_api_key(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        expected = {"ok": True}
        _setup_mock_client(MockClient, "delete", _mock_response(expected))
        result = sdk.delete_api_key("k1")
        assert result == expected


class TestHTTPErrorHandling:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_raises_on_error(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        resp = _mock_response({"detail": "not found"}, status_code=404)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=resp,
        )
        _setup_mock_client(MockClient, "get", resp)
        with pytest.raises(httpx.HTTPStatusError):
            sdk.get_model("nonexistent")

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_post_raises_on_error(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        resp = _mock_response({"detail": "conflict"}, status_code=409)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "409", request=MagicMock(), response=resp,
        )
        _setup_mock_client(MockClient, "post", resp)
        with pytest.raises(httpx.HTTPStatusError):
            sdk.create_model({"name": "dup"})

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_put_raises_on_error(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        resp = _mock_response({"detail": "bad request"}, status_code=400)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400", request=MagicMock(), response=resp,
        )
        _setup_mock_client(MockClient, "put", resp)
        with pytest.raises(httpx.HTTPStatusError):
            sdk.update_model("m1", {"invalid": "field"})

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_delete_raises_on_error(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        resp = _mock_response({"detail": "not found"}, status_code=404)
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=resp,
        )
        _setup_mock_client(MockClient, "delete", resp)
        with pytest.raises(httpx.HTTPStatusError):
            sdk.delete_model("nonexistent")


class TestRequestURLs:
    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_get_url_correct(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        mock_c = _setup_mock_client(MockClient, "get", _mock_response({}))
        sdk.get_model("m1")
        call_args = mock_c.get.call_args
        assert call_args[0][0] == "http://localhost:11444/api/v1/models/m1"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_post_url_correct(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        mock_c = _setup_mock_client(MockClient, "post", _mock_response({}))
        sdk.create_model({"name": "test"})
        call_args = mock_c.post.call_args
        assert call_args[0][0] == "http://localhost:11444/api/v1/models"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_put_url_correct(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        mock_c = _setup_mock_client(MockClient, "put", _mock_response({}))
        sdk.update_model("m1", {"name": "test"})
        call_args = mock_c.put.call_args
        assert call_args[0][0] == "http://localhost:11444/api/v1/models/m1"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_delete_url_correct(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        mock_c = _setup_mock_client(MockClient, "delete", _mock_response({}))
        sdk.delete_model("m1")
        call_args = mock_c.delete.call_args
        assert call_args[0][0] == "http://localhost:11444/api/v1/models/m1"

    @patch("fusion_model_hub.sdk.client.httpx.Client")
    def test_headers_passed(self, MockClient):
        sdk = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
        mock_c = _setup_mock_client(MockClient, "get", _mock_response({}))
        sdk.get_model("m1")
        call_args = mock_c.get.call_args
        assert call_args[1]["headers"]["X-API-Key"] == "test-key"
