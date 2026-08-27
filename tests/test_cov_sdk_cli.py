import json
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_model_hub.sdk.async_client import AsyncFusionModelHubClient
from fusion_model_hub.sdk.client import FusionModelHubClient

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:11444"


def _mock_response(json_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


def _sync_client_with_mock(method, response):
    c = FusionModelHubClient(base_url=BASE_URL, api_key="test-key")
    mock_inner = MagicMock()
    mock_inner.is_closed = False
    getattr(mock_inner, method).return_value = response
    c._client = mock_inner
    return c, mock_inner


def _async_client_with_mock(method, response):
    c = AsyncFusionModelHubClient(base_url=BASE_URL, api_key="test-key")
    mock_inner = AsyncMock()
    mock_inner.is_closed = False
    getattr(mock_inner, method).return_value = response
    c._client = mock_inner
    return c, mock_inner


# ===========================================================================
# sdk/client.py — close/__enter__/__exit__ + hardware/recommend/adapt/
# benchmarks/analyze/ratings/favorites/branches methods (lines 49-57, 316-479)
# ===========================================================================


class TestSyncClientLifecycle:
    def test_close_closes_client(self):
        c = FusionModelHubClient(base_url=BASE_URL)
        mock_inner = MagicMock()
        mock_inner.is_closed = False
        c._client = mock_inner
        c.close()
        mock_inner.close.assert_called_once()
        assert c._client is None

    def test_close_noop_when_no_client(self):
        c = FusionModelHubClient(base_url=BASE_URL)
        c._client = None
        c.close()
        assert c._client is None

    def test_close_noop_when_already_closed(self):
        c = FusionModelHubClient(base_url=BASE_URL)
        mock_inner = MagicMock()
        mock_inner.is_closed = True
        c._client = mock_inner
        c.close()
        mock_inner.close.assert_not_called()

    def test_enter_returns_self(self):
        c = FusionModelHubClient(base_url=BASE_URL)
        result = c.__enter__()
        assert result is c

    def test_exit_closes_client(self):
        c = FusionModelHubClient(base_url=BASE_URL)
        mock_inner = MagicMock()
        mock_inner.is_closed = False
        c._client = mock_inner
        c.__exit__(None, None, None)
        mock_inner.close.assert_called_once()
        assert c._client is None


class TestSyncClientHardware:
    def test_get_hardware_info(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"chip": "m2"}))
        result = c.get_hardware_info()
        assert result == {"chip": "m2"}
        assert mock_inner.get.call_args[0][0].endswith("/api/v1/hardware")

    def test_refresh_hardware(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"ok": True}))
        result = c.refresh_hardware()
        assert result == {"ok": True}
        assert mock_inner.post.call_args[0][0].endswith("/api/v1/hardware/refresh")


class TestSyncClientRecommend:
    def test_recommend_models_default(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"items": []}))
        result = c.recommend_models()
        assert result == {"items": []}
        body = mock_inner.post.call_args[1]["json"]
        assert body["task"] == "llm"
        assert body["preference"] == "balanced"
        assert body["max_results"] == 10
        assert body["min_params_b"] == 0
        assert body["max_params_b"] == 1000

    def test_recommend_models_custom(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"items": []}))
        c.recommend_models(task="vision", preference="speed", max_results=5, min_params=1, max_params=10)
        body = mock_inner.post.call_args[1]["json"]
        assert body["task"] == "vision"
        assert body["preference"] == "speed"
        assert body["max_results"] == 5
        assert body["min_params_b"] == 1
        assert body["max_params_b"] == 10

    def test_quick_recommend(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.quick_recommend(task="code", preference="quality")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"task": "code", "preference": "quality"}


class TestSyncClientAdapt:
    def test_assess_model_minimal(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"score": 0.8}))
        c.assess_model("m1")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"model_id": "m1"}

    def test_assess_model_with_hf_repo_and_format(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"score": 0.9}))
        c.assess_model("m1", hf_repo="org/model", source_format="pytorch")
        body = mock_inner.post.call_args[1]["json"]
        assert body["hf_repo"] == "org/model"
        assert body["source_format"] == "pytorch"

    def test_plan_migration_minimal(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"steps": []}))
        c.plan_migration("m1")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"model_id": "m1", "params_b": 0}

    def test_plan_migration_full(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"steps": []}))
        c.plan_migration("m1", params_b=7.0, hf_repo="org/model", source_format="gguf")
        body = mock_inner.post.call_args[1]["json"]
        assert body["params_b"] == 7.0
        assert body["hf_repo"] == "org/model"
        assert body["source_format"] == "gguf"

    def test_execute_adaptation_minimal(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"execution_id": "e1"}))
        c.execute_adaptation("m1")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"model_id": "m1", "quant_bits": 4, "params_b": 0}

    def test_execute_adaptation_full(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"execution_id": "e2"}))
        c.execute_adaptation("m1", hf_repo="org/m", source_format="safetensors", quant_bits=8, params_b=3.0)
        body = mock_inner.post.call_args[1]["json"]
        assert body["quant_bits"] == 8
        assert body["params_b"] == 3.0
        assert body["hf_repo"] == "org/m"
        assert body["source_format"] == "safetensors"

    def test_get_adapt_execution(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"id": "e1", "status": "done"}))
        result = c.get_adapt_execution("e1")
        assert result["id"] == "e1"
        assert mock_inner.get.call_args[0][0].endswith("/api/v1/adapt/execute/e1")


class TestSyncClientBenchmarks:
    def test_list_benchmarks_no_filters(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_benchmarks()
        assert mock_inner.get.call_args[1]["params"] == {}

    def test_list_benchmarks_all_filters(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_benchmarks(chip="m2", model_id="m1", quant="4bit")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"chip": "m2", "model_id": "m1", "quant": "4bit"}

    def test_get_benchmark_minimal(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"score": 90.0}))
        c.get_benchmark("m1")
        assert mock_inner.get.call_args[1]["params"] == {}

    def test_get_benchmark_with_chip_and_quant(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"score": 90.0}))
        c.get_benchmark("m1", chip="m2", quant="8bit")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"chip": "m2", "quant": "8bit"}


class TestSyncClientAnalyze:
    def test_analyze_model_empty(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"layers": 32}))
        c.analyze_model()
        assert mock_inner.post.call_args[1]["json"] == {}

    def test_analyze_model_with_path(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"layers": 32}))
        c.analyze_model(model_path="/data/model")
        assert mock_inner.post.call_args[1]["json"] == {"model_path": "/data/model"}

    def test_analyze_model_with_hf_repo(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"layers": 32}))
        c.analyze_model(hf_repo="org/model")
        assert mock_inner.post.call_args[1]["json"] == {"hf_repo": "org/model"}

    def test_analyze_model_full(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"layers": 32}))
        c.analyze_model(model_path="/data/model", hf_repo="org/model")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"model_path": "/data/model", "hf_repo": "org/model"}


class TestSyncClientRatings:
    def test_create_rating(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"id": "r1"}))
        c.create_rating("m1", 5, comment="great")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"score": 5, "comment": "great"}
        assert mock_inner.post.call_args[0][0].endswith("/api/v1/models/m1/ratings")

    def test_list_ratings(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_ratings("m1", page=2)
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"page": 2}

    def test_get_rating_summary(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"avg_score": 4.5}))
        result = c.get_rating_summary("m1")
        assert result["avg_score"] == 4.5

    def test_delete_rating(self):
        c, mock_inner = _sync_client_with_mock("delete", _mock_response({"ok": True}))
        c.delete_rating("r1")
        assert mock_inner.delete.call_args[0][0].endswith("/api/v1/models/ratings/r1")


class TestSyncClientFavorites:
    def test_add_favorite(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"id": "f1"}))
        c.add_favorite("m1")
        assert mock_inner.post.call_args[0][0].endswith("/api/v1/models/m1/favorites")

    def test_list_favorites(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_favorites("m1", limit=5)
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"limit": 5}

    def test_list_my_favorites(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_my_favorites()
        assert mock_inner.get.call_args[0][0].endswith("/api/v1/models/favorites/me")

    def test_remove_favorite(self):
        c, mock_inner = _sync_client_with_mock("delete", _mock_response({"ok": True}))
        c.remove_favorite("f1")
        assert mock_inner.delete.call_args[0][0].endswith("/api/v1/models/favorites/f1")


class TestSyncClientBranches:
    def test_create_branch(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"id": "b1"}))
        c.create_branch("m1", "feature-x", base_version_id="v1", description="desc")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"name": "feature-x", "base_version_id": "v1", "description": "desc"}

    def test_list_branches_no_status(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_branches("m1")
        assert mock_inner.get.call_args[1]["params"] == {}

    def test_list_branches_with_status(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_branches("m1", status="active")
        assert mock_inner.get.call_args[1]["params"] == {"status": "active"}

    def test_get_branch(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"id": "b1"}))
        c.get_branch("b1")
        assert mock_inner.get.call_args[0][0].endswith("/api/v1/models/branches/b1")

    def test_update_branch(self):
        c, mock_inner = _sync_client_with_mock("put", _mock_response({"id": "b1"}))
        c.update_branch("b1", {"description": "updated"})
        assert mock_inner.put.call_args[0][0].endswith("/api/v1/models/branches/b1")

    def test_delete_branch(self):
        c, mock_inner = _sync_client_with_mock("delete", _mock_response({"ok": True}))
        c.delete_branch("b1")
        assert mock_inner.delete.call_args[0][0].endswith("/api/v1/models/branches/b1")

    def test_merge_branch(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"merged": True}))
        c.merge_branch("b1")
        assert mock_inner.post.call_args[0][0].endswith("/api/v1/models/branches/b1/merge")


class TestSyncClientLayeredQuantize:
    def test_start_layered_quantize(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"job_id": "j1"}))
        c.start_layered_quantize("m1", default_bits=8, layer_rules=[{"layer": "a", "bits": 4}], output_path="/out")
        body = mock_inner.post.call_args[1]["json"]
        assert body["model"] == "m1"
        assert body["default_bits"] == 8
        assert body["layer_rules"] == [{"layer": "a", "bits": 4}]
        assert body["output_path"] == "/out"

    def test_get_layered_quantize_job(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"id": "j1", "status": "running"}))
        c.get_layered_quantize_job("j1")
        assert mock_inner.get.call_args[0][0].endswith("/api/v1/quantize/layered/jobs/j1")

    def test_list_layered_quantize_jobs(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_layered_quantize_jobs()
        assert mock_inner.get.call_args[0][0].endswith("/api/v1/quantize/layered/jobs")

    def test_evaluate_quantize(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"score": 0.95}))
        c.evaluate_quantize("v1", quant_bits=4, sample_size=64)
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"source_version_id": "v1", "quant_bits": 4, "sample_size": 64}


class TestSyncClientDistributedTask:
    def test_submit_distributed_task_minimal(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"task_id": "t1"}))
        c.submit_distributed_task("benchmark", "v1")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"task_type": "benchmark", "model_version_id": "v1", "config": "{}"}

    def test_submit_distributed_task_with_targets(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"task_id": "t2"}))
        c.submit_distributed_task("inference", "v1", target_node_ids=["n1", "n2"], config='{"k": 1}')
        body = mock_inner.post.call_args[1]["json"]
        assert body["target_node_ids"] == ["n1", "n2"]
        assert body["config"] == '{"k": 1}'

    def test_get_distributed_task(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"id": "t1", "status": "done"}))
        c.get_distributed_task("t1")
        assert mock_inner.get.call_args[0][0].endswith("/api/v1/cluster/distributed-tasks/t1")


class TestSyncClientSystemAuth:
    def test_health(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"status": "ok"}))
        assert c.health() == {"status": "ok"}

    def test_storage_stats(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"free": 100}))
        assert c.storage_stats() == {"free": 100}

    def test_export_data(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"data": "ok"}))
        c.export_data(format="json")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"format": "json"}

    def test_create_api_key(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"id": "k1"}))
        c.create_api_key("my-key")
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"name": "my-key"}

    def test_list_api_keys(self):
        c, mock_inner = _sync_client_with_mock("get", _mock_response({"items": []}))
        c.list_api_keys()

    def test_deactivate_api_key(self):
        c, mock_inner = _sync_client_with_mock("post", _mock_response({"ok": True}))
        c.deactivate_api_key("k1")
        assert mock_inner.post.call_args[0][0].endswith("/api/v1/auth/keys/k1/deactivate")

    def test_delete_api_key(self):
        c, mock_inner = _sync_client_with_mock("delete", _mock_response({"ok": True}))
        c.delete_api_key("k1")
        assert mock_inner.delete.call_args[0][0].endswith("/api/v1/auth/keys/k1")


# ===========================================================================
# sdk/async_client.py — layered quantize + hardware + recommend + adapt +
# benchmarks + analyze (lines 175-191, 362-453)
# ===========================================================================


class TestAsyncClientLayeredQuantize:
    async def test_start_layered_quantize(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"job_id": "j1"}))
        await c.start_layered_quantize("m1", default_bits=8, layer_rules=[{"l": "a"}], output_path="/o")
        body = mock_inner.post.call_args[1]["json"]
        assert body["model"] == "m1"
        assert body["default_bits"] == 8
        assert body["layer_rules"] == [{"l": "a"}]
        assert body["output_path"] == "/o"

    async def test_get_layered_quantize_job(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"id": "j1"}))
        result = await c.get_layered_quantize_job("j1")
        assert result["id"] == "j1"

    async def test_list_layered_quantize_jobs(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"items": []}))
        result = await c.list_layered_quantize_jobs()
        assert "items" in result

    async def test_evaluate_quantize(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"score": 0.9}))
        await c.evaluate_quantize("v1", quant_bits=4, sample_size=32)
        body = mock_inner.post.call_args[1]["json"]
        assert body == {"source_version_id": "v1", "quant_bits": 4, "sample_size": 32}


class TestAsyncClientHardware:
    async def test_get_hardware_info(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"chip": "m2"}))
        result = await c.get_hardware_info()
        assert result == {"chip": "m2"}

    async def test_refresh_hardware(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"ok": True}))
        result = await c.refresh_hardware()
        assert result == {"ok": True}


class TestAsyncClientRecommend:
    async def test_recommend_models(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"items": []}))
        await c.recommend_models(task="code", preference="speed", max_results=3, min_params=1, max_params=20)
        body = mock_inner.post.call_args[1]["json"]
        assert body["task"] == "code"
        assert body["preference"] == "speed"
        assert body["max_results"] == 3
        assert body["min_params_b"] == 1
        assert body["max_params_b"] == 20

    async def test_quick_recommend(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"items": []}))
        await c.quick_recommend(task="llm", preference="balanced")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"task": "llm", "preference": "balanced"}


class TestAsyncClientAdapt:
    async def test_assess_model_minimal(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"score": 0.8}))
        await c.assess_model("m1")
        assert mock_inner.post.call_args[1]["json"] == {"model_id": "m1"}

    async def test_assess_model_full(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"score": 0.8}))
        await c.assess_model("m1", hf_repo="org/m", source_format="pytorch")
        body = mock_inner.post.call_args[1]["json"]
        assert body["hf_repo"] == "org/m"
        assert body["source_format"] == "pytorch"

    async def test_plan_migration_minimal(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"steps": []}))
        await c.plan_migration("m1")
        assert mock_inner.post.call_args[1]["json"] == {"model_id": "m1", "params_b": 0}

    async def test_plan_migration_full(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"steps": []}))
        await c.plan_migration("m1", params_b=7.0, hf_repo="org/m", source_format="gguf")
        body = mock_inner.post.call_args[1]["json"]
        assert body["params_b"] == 7.0
        assert body["hf_repo"] == "org/m"
        assert body["source_format"] == "gguf"

    async def test_execute_adaptation_minimal(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"execution_id": "e1"}))
        await c.execute_adaptation("m1")
        assert mock_inner.post.call_args[1]["json"] == {"model_id": "m1", "quant_bits": 4, "params_b": 0}

    async def test_execute_adaptation_full(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"execution_id": "e2"}))
        await c.execute_adaptation("m1", hf_repo="org/m", source_format="safetensors", quant_bits=8, params_b=3.0)
        body = mock_inner.post.call_args[1]["json"]
        assert body["quant_bits"] == 8
        assert body["params_b"] == 3.0
        assert body["hf_repo"] == "org/m"
        assert body["source_format"] == "safetensors"

    async def test_get_adapt_execution(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"id": "e1"}))
        result = await c.get_adapt_execution("e1")
        assert result["id"] == "e1"


class TestAsyncClientBenchmarks:
    async def test_list_benchmarks_no_filters(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"items": []}))
        await c.list_benchmarks()
        assert mock_inner.get.call_args[1]["params"] == {}

    async def test_list_benchmarks_all_filters(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"items": []}))
        await c.list_benchmarks(chip="m2", model_id="m1", quant="4bit")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"chip": "m2", "model_id": "m1", "quant": "4bit"}

    async def test_get_benchmark_minimal(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"score": 90.0}))
        await c.get_benchmark("m1")
        assert mock_inner.get.call_args[1]["params"] == {}

    async def test_get_benchmark_with_filters(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"score": 90.0}))
        await c.get_benchmark("m1", chip="m2", quant="8bit")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"chip": "m2", "quant": "8bit"}


class TestAsyncClientAnalyze:
    async def test_analyze_model_empty(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"layers": 32}))
        await c.analyze_model()
        assert mock_inner.post.call_args[1]["json"] == {}

    async def test_analyze_model_with_path(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"layers": 32}))
        await c.analyze_model(model_path="/data/m")
        assert mock_inner.post.call_args[1]["json"] == {"model_path": "/data/m"}

    async def test_analyze_model_with_hf_repo(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"layers": 32}))
        await c.analyze_model(hf_repo="org/m")
        assert mock_inner.post.call_args[1]["json"] == {"hf_repo": "org/m"}

    async def test_analyze_model_full(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"layers": 32}))
        await c.analyze_model(model_path="/d/m", hf_repo="org/m")
        assert mock_inner.post.call_args[1]["json"] == {"model_path": "/d/m", "hf_repo": "org/m"}


class TestAsyncClientSystemAuth:
    async def test_health(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"status": "ok"}))
        assert await c.health() == {"status": "ok"}

    async def test_storage_stats(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"free": 100}))
        assert await c.storage_stats() == {"free": 100}

    async def test_export_data(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"data": "ok"}))
        await c.export_data(format="json")
        params = mock_inner.get.call_args[1]["params"]
        assert params == {"format": "json"}

    async def test_create_api_key(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"id": "k1"}))
        await c.create_api_key("my-key")
        assert mock_inner.post.call_args[1]["json"] == {"name": "my-key"}

    async def test_list_api_keys(self):
        c, mock_inner = _async_client_with_mock("get", _mock_response({"items": []}))
        await c.list_api_keys()

    async def test_deactivate_api_key(self):
        c, mock_inner = _async_client_with_mock("post", _mock_response({"ok": True}))
        await c.deactivate_api_key("k1")
        assert mock_inner.post.call_args[0][0].endswith("/api/v1/auth/keys/k1/deactivate")

    async def test_delete_api_key(self):
        c, mock_inner = _async_client_with_mock("delete", _mock_response({"ok": True}))
        await c.delete_api_key("k1")
        assert mock_inner.delete.call_args[0][0].endswith("/api/v1/auth/keys/k1")


# ===========================================================================
# cli/list_cmd.py — list_local / list_remote / list_stats (lines 18-77)
# ===========================================================================


class TestCliListLocal:
    def test_list_local_no_models(self, tmp_path):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app

        runner = CliRunner()
        result = runner.invoke(list_app, ["local", "--storage-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No local models" in result.output

    def test_list_local_with_models(self, tmp_path):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app
        from fusion_model_hub.manage.manager import LocalModelManager

        manager = LocalModelManager(str(tmp_path))
        manager.register("model-1", "Test Model", str(tmp_path / "model1.bin"))
        runner = CliRunner()
        result = runner.invoke(list_app, ["local", "--storage-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "model-1" in result.output
        assert "Test Model" in result.output

    def test_list_local_with_active_and_size(self, tmp_path):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app
        from fusion_model_hub.manage.manager import LocalModelManager

        model_file = tmp_path / "big.bin"
        model_file.write_bytes(b"x" * (2 * 1024 * 1024 * 1024 // 1))  # ~2GB equivalent bytes
        manager = LocalModelManager(str(tmp_path))
        manager.register("m-active", "Active Model", str(model_file))
        manager.set_active("m-active")
        runner = CliRunner()
        result = runner.invoke(list_app, ["local", "--storage-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "*" in result.output
        assert "m-active" in result.output

    def test_list_local_with_bad_path(self, tmp_path):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app
        from fusion_model_hub.manage.manager import LocalModelManager

        manager = LocalModelManager(str(tmp_path))
        manager.register("m-bad", "Bad Path Model", "/nonexistent/path/model.bin")
        runner = CliRunner()
        result = runner.invoke(list_app, ["local", "--storage-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "m-bad" in result.output


class TestCliListRemote:
    def test_list_remote_success_list(self):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"id": "m1", "name": "Remote Model", "params_b": 7.0},
        ]

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get.return_value = mock_resp
            MockClient.return_value = mock_instance
            runner = CliRunner()
            result = runner.invoke(list_app, ["remote", "--limit", "5"])
        assert result.exit_code == 0
        assert "Remote Model" in result.output

    def test_list_remote_success_dict_envelope(self):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": [{"id": "m2", "name": "Env Model", "params_b": 3.0}]}

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get.return_value = mock_resp
            MockClient.return_value = mock_instance
            runner = CliRunner()
            result = runner.invoke(list_app, ["remote"])
        assert result.exit_code == 0
        assert "Env Model" in result.output

    def test_list_remote_non_list_response(self):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": "weird"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get.return_value = mock_resp
            MockClient.return_value = mock_instance
            runner = CliRunner()
            result = runner.invoke(list_app, ["remote"])
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_list_remote_connection_error(self):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get.side_effect = Exception("connection refused")
            MockClient.return_value = mock_instance
            runner = CliRunner()
            result = runner.invoke(list_app, ["remote"])
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_list_remote_non_200(self):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            mock_instance.get.return_value = mock_resp
            MockClient.return_value = mock_instance
            runner = CliRunner()
            result = runner.invoke(list_app, ["remote"])
        assert result.exit_code == 0


class TestCliListStats:
    def test_list_stats_empty(self, tmp_path):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app

        runner = CliRunner()
        result = runner.invoke(list_app, ["stats", "--storage-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Total models: 0" in result.output
        assert "Active models: 0" in result.output

    def test_list_stats_with_models(self, tmp_path):
        from typer.testing import CliRunner

        from fusion_model_hub.cli.list_cmd import list_app
        from fusion_model_hub.manage.manager import LocalModelManager

        model_file = tmp_path / "model.bin"
        model_file.write_bytes(b"x" * 2048)
        manager = LocalModelManager(str(tmp_path))
        manager.register("m1", "Model One", str(model_file))
        manager.set_active("m1")
        runner = CliRunner()
        result = runner.invoke(list_app, ["stats", "--storage-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Total models: 1" in result.output
        assert "Active models: 1" in result.output


# ===========================================================================
# server/__main__.py — _run_import (stdin, invalid type, tags),
# _run_restore, _run_migrate (alembic installed), TLS serve branch,
# subcommand dispatch (lines 87, 104-105, 116, 133-145, 149-167, 230-232,
# 241-248)
# ===========================================================================


class TestMainImport:
    def test_import_from_stdin(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_import

        input_data = {
            "models": [{"name": "stdin-model", "model_type": "llm"}],
            "tenants": [],
            "webhooks": [],
        }

        class Args:
            data_dir = str(tmp_path / "stdin_import")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = "-"

        with patch("sys.stdin", MagicMock(read=lambda: json.dumps(input_data))):
            _run_import(Args())

    def test_import_invalid_model_type_falls_back(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_import

        input_data = {
            "models": [{"name": "bad-type-model", "model_type": "not_a_real_type"}],
            "tenants": [],
            "webhooks": [],
        }
        input_file = tmp_path / "bad_type.json"
        input_file.write_text(json.dumps(input_data))

        class Args:
            data_dir = str(tmp_path / "bad_type_dir")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(input_file)

        _run_import(Args())

    def test_import_with_tags(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_import

        input_data = {
            "models": [
                {
                    "name": "tagged-model",
                    "model_type": "llm",
                    "tags": [{"key": "family", "value": "qwen"}],
                }
            ],
            "tenants": [],
            "webhooks": [],
        }
        input_file = tmp_path / "tagged.json"
        input_file.write_text(json.dumps(input_data))

        class Args:
            data_dir = str(tmp_path / "tagged_dir")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(input_file)

        _run_import(Args())

    def test_import_with_tenants_and_webhooks(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_import

        input_data = {
            "models": [{"name": "m-with-ten", "model_type": "llm", "tenant_id": "t1"}],
            "tenants": [{"name": "tenant-1", "display_name": "Tenant One"}],
            "webhooks": [{"name": "wh-1", "url": "http://example.com/hook", "events": "model.published"}],
        }
        input_file = tmp_path / "ten_wh.json"
        input_file.write_text(json.dumps(input_data))

        class Args:
            data_dir = str(tmp_path / "ten_wh_dir")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(input_file)

        _run_import(Args())


class TestMainExport:
    def test_export_to_stdout(self, tmp_path, capsys):
        from fusion_model_hub.server.__main__ import _run_export

        class Args:
            data_dir = str(tmp_path / "stdout_export")
            db_url = "sqlite+aiosqlite:///:memory:"
            output = "-"
            models = ""

        _run_export(Args())
        captured = capsys.readouterr()
        assert '"version": "1.0"' in captured.out

    def test_export_with_model_filter(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_export

        out_file = tmp_path / "filtered.json"

        class Args:
            data_dir = str(tmp_path / "filter_export")
            db_url = "sqlite+aiosqlite:///:memory:"
            output = str(out_file)
            models = "nonexistent-id"

        _run_export(Args())
        assert os.path.exists(out_file)
        with open(out_file) as f:
            data = json.load(f)
        assert data["models"] == []


class TestMainMigrateImportError:
    def test_migrate_without_alembic_exits(self):
        from fusion_model_hub.server.__main__ import _run_migrate

        class Args:
            db_url = "sqlite:///:memory:"
            revision = ""

        real_alembic = sys.modules.get("alembic")
        real_alembic_config = sys.modules.get("alembic.config")
        real_alembic_command = sys.modules.get("alembic.command")
        removed = {}
        for name in ("alembic", "alembic.config", "alembic.command"):
            if name in sys.modules:
                removed[name] = sys.modules.pop(name)
        sys.modules["alembic"] = None
        sys.modules["alembic.config"] = None
        sys.modules["alembic.command"] = None
        try:
            with pytest.raises(SystemExit) as exc_info:
                _run_migrate(Args())
            assert exc_info.value.code == 1
        finally:
            for name in ("alembic", "alembic.config", "alembic.command"):
                sys.modules[name] = removed.get(name)
            if real_alembic is not None:
                sys.modules["alembic"] = real_alembic
            if real_alembic_config is not None:
                sys.modules["alembic.config"] = real_alembic_config
            if real_alembic_command is not None:
                sys.modules["alembic.command"] = real_alembic_command


class TestMainRestore:
    def test_restore_from_backup_file(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_restore

        backup_file = tmp_path / "backup.json"
        backup_data = {
            "models": [
                {
                    "id": "restored-m1",
                    "name": "Restored Model",
                    "model_type": "llm",
                    "architecture": "qwen2",
                    "params_size": "7B",
                    "license": "apache-2.0",
                }
            ],
            "versions": [
                {
                    "id": "restored-v1",
                    "model_id": "restored-m1",
                    "version": "1.0.0",
                    "format": "mlx",
                    "quantization": "4bit",
                    "status": "published",
                    "file_hash": "abc123",
                    "file_size": 1024,
                    "benchmark_score": 85.0,
                }
            ],
        }
        backup_file.write_text(json.dumps(backup_data))

        class Args:
            data_dir = str(tmp_path / "restore_data")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(backup_file)

        _run_restore(Args())

    def test_restore_empty_backup(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_restore

        backup_file = tmp_path / "empty_backup.json"
        backup_file.write_text(json.dumps({"models": [], "versions": []}))

        class Args:
            data_dir = str(tmp_path / "restore_empty")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(backup_file)

        _run_restore(Args())

    def test_restore_invalid_enum_falls_back(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_restore

        backup_file = tmp_path / "bad_enum.json"
        backup_data = {
            "models": [
                {
                    "id": "m-enum",
                    "name": "Enum Model",
                    "model_type": "not_valid",
                }
            ],
            "versions": [
                {
                    "id": "v-enum",
                    "model_id": "m-enum",
                    "version": "1.0.0",
                    "format": "not_a_format",
                    "quantization": "not_a_quant",
                    "status": "not_a_status",
                }
            ],
        }
        backup_file.write_text(json.dumps(backup_data))

        class Args:
            data_dir = str(tmp_path / "restore_enum")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(backup_file)

        _run_restore(Args())

    def test_restore_skips_existing_ids(self, tmp_path):
        from fusion_model_hub.server.__main__ import _run_restore

        backup_file = tmp_path / "dup_backup.json"
        backup_data = {
            "models": [{"id": "dup-m1", "name": "Dup Model", "model_type": "llm"}],
            "versions": [],
        }
        backup_file.write_text(json.dumps(backup_data))

        class Args:
            data_dir = str(tmp_path / "dup_restore")
            db_url = "sqlite+aiosqlite:///:memory:"
            input = str(backup_file)

        _run_restore(Args())
        _run_restore(Args())


class TestMainMigrate:
    def test_migrate_with_alembic_installed(self):
        from fusion_model_hub.server.__main__ import _run_migrate

        class Args:
            db_url = "sqlite:///:memory:"
            revision = ""

        mock_config = MagicMock()
        mock_command = MagicMock()
        alembic_mod = MagicMock()
        alembic_mod.command = mock_command

        with patch.dict(
            sys.modules,
            {
                "alembic": alembic_mod,
                "alembic.config": MagicMock(Config=mock_config),
                "alembic.command": mock_command,
            },
        ):
            _run_migrate(Args())
            mock_command.upgrade.assert_called_once()

    def test_migrate_with_revision(self):
        from fusion_model_hub.server.__main__ import _run_migrate

        class Args:
            db_url = "sqlite:///:memory:"
            revision = "abc123"

        mock_config = MagicMock()
        mock_command = MagicMock()
        alembic_mod = MagicMock()
        alembic_mod.command = mock_command

        with patch.dict(
            sys.modules,
            {
                "alembic": alembic_mod,
                "alembic.config": MagicMock(Config=mock_config),
                "alembic.command": mock_command,
            },
        ):
            _run_migrate(Args())
            mock_command.upgrade.assert_called_once_with(mock_config.return_value, "abc123")


class TestMainServeDispatch:
    def test_main_serve_with_tls(self):
        from fusion_model_hub.server.__main__ import main

        with patch(
            "sys.argv",
            [
                "fusion-model-hub",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "9998",
                "--tls-certfile",
                "/tmp/cert.pem",
                "--tls-keyfile",
                "/tmp/key.pem",
            ],
        ):
            with patch("uvicorn.run") as mock_run:
                main()
                mock_run.assert_called_once()
                kwargs = mock_run.call_args[1]
                assert kwargs["ssl_certfile"] == "/tmp/cert.pem"
                assert kwargs["ssl_keyfile"] == "/tmp/key.pem"

    def test_main_export_dispatch(self, tmp_path):
        from fusion_model_hub.server.__main__ import main

        out_file = str(tmp_path / "dispatch_export.json")
        with patch(
            "sys.argv",
            [
                "fusion-model-hub",
                "export",
                "--data-dir",
                str(tmp_path / "dispatch_data"),
                "--db-url",
                "sqlite+aiosqlite:///:memory:",
                "--output",
                out_file,
            ],
        ):
            main()
        assert os.path.exists(out_file)

    def test_main_import_dispatch(self, tmp_path):
        from fusion_model_hub.server.__main__ import main

        input_file = tmp_path / "dispatch_import.json"
        input_file.write_text(
            json.dumps(
                {
                    "models": [{"name": "dispatch-model", "model_type": "llm"}],
                    "tenants": [],
                    "webhooks": [],
                }
            )
        )
        with patch(
            "sys.argv",
            [
                "fusion-model-hub",
                "import",
                "--data-dir",
                str(tmp_path / "dispatch_import_dir"),
                "--db-url",
                "sqlite+aiosqlite:///:memory:",
                "--input",
                str(input_file),
            ],
        ):
            main()

    def test_main_migrate_dispatch(self):
        from fusion_model_hub.server.__main__ import main

        with patch("sys.argv", ["fusion-model-hub", "migrate", "--db-url", "sqlite:///:memory:"]):
            mock_config = MagicMock()
            mock_command = MagicMock()
            alembic_mod = MagicMock()
            alembic_mod.command = mock_command
            with patch.dict(
                sys.modules,
                {
                    "alembic": alembic_mod,
                    "alembic.config": MagicMock(Config=mock_config),
                    "alembic.command": mock_command,
                },
            ):
                main()
                mock_command.upgrade.assert_called_once()

    def test_main_restore_dispatch(self, tmp_path):
        from fusion_model_hub.server.__main__ import main

        backup_file = tmp_path / "dispatch_backup.json"
        backup_file.write_text(
            json.dumps(
                {
                    "models": [{"id": "dm1", "name": "Dispatch Model", "model_type": "llm"}],
                    "versions": [],
                }
            )
        )
        with patch(
            "sys.argv",
            [
                "fusion-model-hub",
                "restore",
                "--data-dir",
                str(tmp_path / "dispatch_restore_dir"),
                "--db-url",
                "sqlite+aiosqlite:///:memory:",
                "--input",
                str(backup_file),
            ],
        ):
            main()
