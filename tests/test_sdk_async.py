import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_model_hub.sdk.async_client import AsyncFusionModelHubClient

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:11444"


def _mock_response(json_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def client():
    return AsyncFusionModelHubClient(base_url=BASE_URL, api_key="test-key")


class TestAsyncClientInit:
    def test_base_url_strips_trailing_slash(self):
        c = AsyncFusionModelHubClient(base_url="http://host:9999/")
        assert c._base_url == "http://host:9999"

    def test_api_key_header(self):
        c = AsyncFusionModelHubClient(api_key="mykey")
        assert c._headers["X-API-Key"] == "mykey"

    def test_no_api_key_header(self):
        c = AsyncFusionModelHubClient()
        assert "X-API-Key" not in c._headers

    def test_url_construction(self):
        c = AsyncFusionModelHubClient(base_url="http://host:11444")
        assert c._url("/models") == "http://host:11444/api/v1/models"

    def test_default_timeout(self):
        c = AsyncFusionModelHubClient()
        assert c._timeout == 30.0


class TestAsyncClientContextManager:
    @pytest.mark.asyncio
    async def test_aenter_returns_self(self, client):
        result = await client.__aenter__()
        assert result is client

    @pytest.mark.asyncio
    async def test_aexit_closes_client(self, client):
        mock_inner = AsyncMock()
        mock_inner.is_closed = False
        client._client = mock_inner
        await client.__aexit__(None, None, None)
        mock_inner.aclose.assert_awaited_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_aexit_noop_if_no_client(self, client):
        client._client = None
        await client.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_close_skips_if_already_closed(self, client):
        mock_inner = AsyncMock()
        mock_inner.is_closed = True
        client._client = mock_inner
        await client.close()
        mock_inner.aclose.assert_not_awaited()


class TestAsyncClientGetClient:
    @pytest.mark.asyncio
    async def test_creates_client_on_first_call(self, client):
        assert client._client is None
        with patch("fusion_model_hub.sdk.async_client.httpx.AsyncClient") as MockAsync:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            MockAsync.return_value = mock_instance
            c = await client._get_client()
            assert c is mock_instance
            MockAsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_existing_client(self, client):
        mock_inner = AsyncMock()
        mock_inner.is_closed = False
        client._client = mock_inner
        c = await client._get_client()
        assert c is mock_inner

    @pytest.mark.asyncio
    async def test_recreates_closed_client(self, client):
        mock_old = AsyncMock()
        mock_old.is_closed = True
        client._client = mock_old
        with patch("fusion_model_hub.sdk.async_client.httpx.AsyncClient") as MockAsync:
            mock_new = AsyncMock()
            mock_new.is_closed = False
            MockAsync.return_value = mock_new
            c = await client._get_client()
            assert c is mock_new


class TestAsyncModelsMethods:
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_models(page=1)
        mock_inner.get.assert_awaited_once()
        assert "items" in result

    @pytest.mark.asyncio
    async def test_get_model(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "m1", "name": "test"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_model("m1")
        assert result["id"] == "m1"

    @pytest.mark.asyncio
    async def test_create_model(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "m1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.create_model({"name": "test"})
        assert result["id"] == "m1"

    @pytest.mark.asyncio
    async def test_update_model(self, client):
        mock_inner = AsyncMock()
        mock_inner.put.return_value = _mock_response({"id": "m1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.update_model("m1", {"description": "updated"})
        assert result["id"] == "m1"

    @pytest.mark.asyncio
    async def test_delete_model(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.delete_model("m1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_import_from_hf(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "m2"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.import_from_hf({"hf_repo": "org/model"})
        assert result["id"] == "m2"

    @pytest.mark.asyncio
    async def test_sync_registry(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"synced": 3})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.sync_registry("http://registry.example.com")
        call_args = mock_inner.post.call_args
        assert "sync" in call_args[0][0]
        assert result["synced"] == 3

    @pytest.mark.asyncio
    async def test_batch_delete(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"deleted": 2})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.batch_delete(["m1", "m2"])
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["model_ids"] == ["m1", "m2"]

    @pytest.mark.asyncio
    async def test_compare_models(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"comparison": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.compare_models(["m1", "m2"])
        call_args = mock_inner.get.call_args
        assert "model_ids" in call_args[1]["params"]


class TestAsyncVersionsMethods:
    @pytest.mark.asyncio
    async def test_list_versions(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_versions("m1")
        assert "items" in result

    @pytest.mark.asyncio
    async def test_get_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "v1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_version("v1")
        assert result["id"] == "v1"

    @pytest.mark.asyncio
    async def test_update_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.put.return_value = _mock_response({"id": "v1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.update_version("v1", {"release_notes": "updated"})
        assert result["id"] == "v1"

    @pytest.mark.asyncio
    async def test_delete_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.delete_version("v1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_promote_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "published"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.promote_version("v1")
        assert result["status"] == "published"

    @pytest.mark.asyncio
    async def test_benchmark_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"benchmark_score": 95.0})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.benchmark_version("v1")
        assert result["benchmark_score"] == 95.0

    @pytest.mark.asyncio
    async def test_rollback_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "published"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.rollback_version("v1")
        call_args = mock_inner.post.call_args
        assert "rollback" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_deprecate_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "deprecated"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.deprecate_version("v1")
        call_args = mock_inner.post.call_args
        assert "deprecate" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_retire_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "retired"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.retire_version("v1")
        call_args = mock_inner.post.call_args
        assert "retire" in call_args[0][0]


class TestAsyncQuantizeMethods:
    @pytest.mark.asyncio
    async def test_start_quantize(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "q1", "status": "running"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.start_quantize("v1", target_format="mlx", quant_bits=4)
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["source_version_id"] == "v1"
        assert body["quant_bits"] == 4

    @pytest.mark.asyncio
    async def test_start_quantize_with_calibration(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "q2"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.start_quantize("v1", calibration_dataset="data.jsonl")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["calibration_dataset"] == "data.jsonl"

    @pytest.mark.asyncio
    async def test_list_quantize_tasks(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_quantize_tasks()
        assert "items" in result

    @pytest.mark.asyncio
    async def test_get_quantize_status(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "q1", "status": "completed"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_quantize_status("q1")
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_start_lora_merge(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "l1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.start_lora_merge("v1", "v2")
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["base_version_id"] == "v1"
        assert body["lora_version_id"] == "v2"

    @pytest.mark.asyncio
    async def test_get_lora_merge_status(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "l1", "status": "completed"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_lora_merge_status("l1")
        assert result["status"] == "completed"


class TestAsyncInferenceMethods:
    @pytest.mark.asyncio
    async def test_chat_completions(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"choices": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.chat_completions("m1", [{"role": "user", "content": "hi"}])
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["model"] == "m1"
        assert len(body["messages"]) == 1

    @pytest.mark.asyncio
    async def test_chat_completions_with_kwargs(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"choices": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.chat_completions("m1", [], temperature=0.7)
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_completions(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"choices": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.completions("m1", "hello world")
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["model"] == "m1"
        assert body["prompt"] == "hello world"

    @pytest.mark.asyncio
    async def test_embeddings_single(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"data": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.embeddings("m1", "hello")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["input"] == "hello"

    @pytest.mark.asyncio
    async def test_embeddings_list(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"data": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.embeddings("m1", ["hello", "world"])
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["input"] == ["hello", "world"]


class TestAsyncSecurityMethods:
    @pytest.mark.asyncio
    async def test_start_security_scan(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "s1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.start_security_scan("v1")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["version_id"] == "v1"

    @pytest.mark.asyncio
    async def test_start_security_scan_custom_type(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "s2"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.start_security_scan("v1", scan_type="quick")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["scan_type"] == "quick"

    @pytest.mark.asyncio
    async def test_get_security_scan(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "s1", "status": "completed"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_security_scan("s1")
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_security_scans(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_security_scans()
        assert "items" in result


class TestAsyncWatermarkMethods:
    @pytest.mark.asyncio
    async def test_embed_watermark(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "w1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.embed_watermark("v1")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["version_id"] == "v1"

    @pytest.mark.asyncio
    async def test_embed_watermark_with_metadata(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "w2"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.embed_watermark("v1", metadata='{"owner":"acme"}')
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["metadata"] == '{"owner":"acme"}'

    @pytest.mark.asyncio
    async def test_verify_watermark(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"verified": True})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.verify_watermark("v1")
        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_list_watermarks(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_watermarks()
        assert "items" in result


class TestAsyncEncryptionMethods:
    @pytest.mark.asyncio
    async def test_encrypt_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "encrypted"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.encrypt_version("v1")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["version_id"] == "v1"

    @pytest.mark.asyncio
    async def test_decrypt_version(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "decrypted"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.decrypt_version("v1")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["version_id"] == "v1"

    @pytest.mark.asyncio
    async def test_get_encryption_status(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"encrypted": True})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_encryption_status("v1")
        assert result["encrypted"] is True


class TestAsyncApprovalsMethods:
    @pytest.mark.asyncio
    async def test_create_approval(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "a1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.create_approval("v1", level="L2", reason="deploy")
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["version_id"] == "v1"
        assert body["level"] == "L2"

    @pytest.mark.asyncio
    async def test_list_approvals(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_approvals()
        assert "items" in result

    @pytest.mark.asyncio
    async def test_get_approval(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "a1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_approval("a1")
        assert result["id"] == "a1"

    @pytest.mark.asyncio
    async def test_approve_request(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "approved"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.approve_request("a1", comment="looks good")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["comment"] == "looks good"

    @pytest.mark.asyncio
    async def test_reject_request(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"status": "rejected"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.reject_request("a1", comment="not ready")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["comment"] == "not ready"


class TestAsyncGitLFSMethods:
    @pytest.mark.asyncio
    async def test_gitlfs_batch(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"objects": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.gitlfs_batch("download", [{"oid": "abc"}])
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["operation"] == "download"
        assert len(body["objects"]) == 1

    @pytest.mark.asyncio
    async def test_create_gitlfs_lock(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "l1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.create_gitlfs_lock("/data/model.bin")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["path"] == "/data/model.bin"

    @pytest.mark.asyncio
    async def test_list_gitlfs_locks(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"locks": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_gitlfs_locks()
        assert "locks" in result

    @pytest.mark.asyncio
    async def test_delete_gitlfs_lock(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.delete_gitlfs_lock("l1")
        assert result == {}


class TestAsyncClusterMethods:
    @pytest.mark.asyncio
    async def test_list_nodes(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"nodes": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_nodes()
        assert "nodes" in result

    @pytest.mark.asyncio
    async def test_add_node(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "n1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.add_node("node1", "http://node1:11444")
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["name"] == "node1"
        assert body["url"] == "http://node1:11444"

    @pytest.mark.asyncio
    async def test_add_node_custom_capabilities(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "n2"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.add_node("node2", "http://node2:11444", capabilities="inference")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["capabilities"] == "inference"

    @pytest.mark.asyncio
    async def test_get_node(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "n1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_node("n1")
        assert result["id"] == "n1"

    @pytest.mark.asyncio
    async def test_remove_node(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.remove_node("n1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_submit_distributed_task(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "dt1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.submit_distributed_task("quantize", "v1")
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["task_type"] == "quantize"
        assert body["model_version_id"] == "v1"
        assert "target_node_ids" not in body

    @pytest.mark.asyncio
    async def test_submit_distributed_task_with_targets(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "dt2"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.submit_distributed_task("quantize", "v1", target_node_ids=["n1", "n2"])
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["target_node_ids"] == ["n1", "n2"]

    @pytest.mark.asyncio
    async def test_get_distributed_task(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "dt1", "status": "completed"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_distributed_task("dt1")
        assert result["status"] == "completed"


class TestAsyncRatingsMethods:
    @pytest.mark.asyncio
    async def test_create_rating(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "r1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.create_rating("m1", 5, comment="excellent")
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["score"] == 5
        assert body["comment"] == "excellent"

    @pytest.mark.asyncio
    async def test_list_ratings(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_ratings("m1")
        assert "items" in result

    @pytest.mark.asyncio
    async def test_get_rating_summary(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"avg_score": 4.5})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_rating_summary("m1")
        assert result["avg_score"] == 4.5

    @pytest.mark.asyncio
    async def test_delete_rating(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.delete_rating("r1")
        assert result == {}


class TestAsyncFavoritesMethods:
    @pytest.mark.asyncio
    async def test_add_favorite(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "f1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.add_favorite("m1")
        call_args = mock_inner.post.call_args
        assert "m1/favorites" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_list_favorites(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_favorites("m1")
        assert "items" in result

    @pytest.mark.asyncio
    async def test_list_my_favorites(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_my_favorites()
        call_args = mock_inner.get.call_args
        assert "favorites/me" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_remove_favorite(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.remove_favorite("f1")
        assert result == {}


class TestAsyncBranchesMethods:
    @pytest.mark.asyncio
    async def test_create_branch(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "b1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.create_branch("m1", "feature-x", base_version_id="v1")
        call_args = mock_inner.post.call_args
        body = call_args[1]["json"]
        assert body["name"] == "feature-x"
        assert body["base_version_id"] == "v1"

    @pytest.mark.asyncio
    async def test_list_branches(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_branches("m1")
        assert "items" in result

    @pytest.mark.asyncio
    async def test_list_branches_with_status(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_branches("m1", status="active")
        call_args = mock_inner.get.call_args
        assert call_args[1]["params"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_list_branches_no_status(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_branches("m1")
        call_args = mock_inner.get.call_args
        assert call_args[1]["params"] == {}

    @pytest.mark.asyncio
    async def test_get_branch(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "b1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.get_branch("b1")
        assert result["id"] == "b1"

    @pytest.mark.asyncio
    async def test_update_branch(self, client):
        mock_inner = AsyncMock()
        mock_inner.put.return_value = _mock_response({"id": "b1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.update_branch("b1", {"description": "updated"})
        assert result["id"] == "b1"

    @pytest.mark.asyncio
    async def test_delete_branch(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.delete_branch("b1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_merge_branch(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"merged": True})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.merge_branch("b1")
        call_args = mock_inner.post.call_args
        assert "merge" in call_args[0][0]


class TestAsyncSystemMethods:
    @pytest.mark.asyncio
    async def test_health(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"status": "healthy"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.health()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_storage_stats(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"total_size": 1024})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.storage_stats()
        assert result["total_size"] == 1024

    @pytest.mark.asyncio
    async def test_export_data(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"data": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.export_data(format="json")
        call_args = mock_inner.get.call_args
        assert call_args[1]["params"]["format"] == "json"


class TestAsyncAuthMethods:
    @pytest.mark.asyncio
    async def test_create_api_key(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "k1", "key": "sk-xxx"})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.create_api_key("mykey")
        call_args = mock_inner.post.call_args
        assert call_args[1]["json"]["name"] == "mykey"

    @pytest.mark.asyncio
    async def test_list_api_keys(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"items": []})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.list_api_keys()
        assert "items" in result

    @pytest.mark.asyncio
    async def test_deactivate_api_key(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"active": False})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.deactivate_api_key("k1")
        call_args = mock_inner.post.call_args
        assert "deactivate" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_delete_api_key(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        result = await client.delete_api_key("k1")
        assert result == {}


class TestAsyncClientURLConstruction:
    @pytest.mark.asyncio
    async def test_get_url_correct(self, client):
        mock_inner = AsyncMock()
        mock_inner.get.return_value = _mock_response({"id": "m1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        await client.get_model("m1")
        call_args = mock_inner.get.call_args
        assert call_args[0][0] == "http://localhost:11444/api/v1/models/m1"

    @pytest.mark.asyncio
    async def test_post_url_correct(self, client):
        mock_inner = AsyncMock()
        mock_inner.post.return_value = _mock_response({"id": "m1"})
        mock_inner.is_closed = False
        client._client = mock_inner
        await client.create_model({"name": "test"})
        call_args = mock_inner.post.call_args
        assert call_args[0][0] == "http://localhost:11444/api/v1/models"

    @pytest.mark.asyncio
    async def test_delete_url_correct(self, client):
        mock_inner = AsyncMock()
        mock_inner.delete.return_value = _mock_response({})
        mock_inner.is_closed = False
        client._client = mock_inner
        await client.delete_model("m1")
        call_args = mock_inner.delete.call_args
        assert call_args[0][0] == "http://localhost:11444/api/v1/models/m1"

    @pytest.mark.asyncio
    async def test_headers_passed_to_client(self, client):
        with patch("fusion_model_hub.sdk.async_client.httpx.AsyncClient") as MockAsync:
            mock_instance = AsyncMock()
            mock_instance.is_closed = False
            MockAsync.return_value = mock_instance
            c = await client._get_client()
            MockAsync.assert_called_once()
            call_kwargs = MockAsync.call_args[1]
            assert call_kwargs["headers"]["X-API-Key"] == "test-key"
