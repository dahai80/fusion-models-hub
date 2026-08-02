import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AsyncFusionModelHubClient:
    def __init__(self, base_url: str = "http://localhost:11444", api_key: str | None = None, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _url(self, path: str) -> str:
        return f"{self._base_url}/api/v1{path}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=self._headers)
        return self._client

    async def _get(self, path: str, params: dict | None = None) -> dict:
        c = await self._get_client()
        r = await c.get(self._url(path), params=params)
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, json: dict | None = None, **kwargs: Any) -> dict:
        c = await self._get_client()
        r = await c.post(self._url(path), json=json, **kwargs)
        r.raise_for_status()
        return r.json()

    async def _put(self, path: str, json: dict | None = None) -> dict:
        c = await self._get_client()
        r = await c.put(self._url(path), json=json)
        r.raise_for_status()
        return r.json()

    async def _delete(self, path: str) -> dict:
        c = await self._get_client()
        r = await c.delete(self._url(path))
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # --- Models ---
    async def list_models(self, **params: Any) -> dict:
        return await self._get("/models", params=params)

    async def get_model(self, model_id: str) -> dict:
        return await self._get(f"/models/{model_id}")

    async def create_model(self, data: dict) -> dict:
        return await self._post("/models", json=data)

    async def update_model(self, model_id: str, data: dict) -> dict:
        return await self._put(f"/models/{model_id}", json=data)

    async def delete_model(self, model_id: str) -> dict:
        return await self._delete(f"/models/{model_id}")

    async def import_from_hf(self, data: dict) -> dict:
        return await self._post("/models/import/hf", json=data)

    async def sync_registry(self, source_url: str) -> dict:
        return await self._post("/models/sync", json={"source_url": source_url})

    async def batch_delete(self, model_ids: list[str]) -> dict:
        return await self._post("/models/batch-delete", json={"model_ids": model_ids})

    async def compare_models(self, model_ids: list[str]) -> dict:
        return await self._get("/models/compare", params={"model_ids": ",".join(model_ids)})

    # --- Versions ---
    async def list_versions(self, model_id: str, **params: Any) -> dict:
        return await self._get(f"/models/{model_id}/versions", params=params)

    async def get_version(self, version_id: str) -> dict:
        return await self._get(f"/versions/{version_id}")

    async def update_version(self, version_id: str, data: dict) -> dict:
        return await self._put(f"/versions/{version_id}", json=data)

    async def delete_version(self, version_id: str) -> dict:
        return await self._delete(f"/versions/{version_id}")

    async def promote_version(self, version_id: str) -> dict:
        return await self._post(f"/versions/{version_id}/promote")

    async def benchmark_version(self, version_id: str) -> dict:
        return await self._post(f"/versions/{version_id}/benchmark")

    async def rollback_version(self, version_id: str) -> dict:
        return await self._post(f"/versions/{version_id}/rollback")

    async def deprecate_version(self, version_id: str) -> dict:
        return await self._post(f"/versions/{version_id}/deprecate")

    async def retire_version(self, version_id: str) -> dict:
        return await self._post(f"/versions/{version_id}/retire")

    # --- Quantize ---
    async def start_quantize(
        self, source_version_id: str, target_format: str = "mlx",
        quant_bits: int = 4, calibration_dataset: str = "",
    ) -> dict:
        return await self._post("/quantize", json={
            "source_version_id": source_version_id,
            "target_format": target_format,
            "quant_bits": quant_bits,
            "calibration_dataset": calibration_dataset,
        })

    async def list_quantize_tasks(self, **params: Any) -> dict:
        return await self._get("/quantize", params=params)

    async def get_quantize_status(self, task_id: str) -> dict:
        return await self._get(f"/quantize/{task_id}")

    async def start_lora_merge(
        self, base_version_id: str, lora_version_id: str,
        target_format: str = "mlx", quant_bits: int = 4,
    ) -> dict:
        return await self._post("/quantize/lora-merge", json={
            "base_version_id": base_version_id,
            "lora_version_id": lora_version_id,
            "target_format": target_format,
            "quant_bits": quant_bits,
        })

    async def get_lora_merge_status(self, task_id: str) -> dict:
        return await self._get(f"/quantize/lora-merge/{task_id}")

    async def start_layered_quantize(
        self, model: str, default_bits: int = 4,
        layer_rules: list[dict] | None = None, output_path: str = "",
    ) -> dict:
        return await self._post("/quantize/layered", json={
            "model": model,
            "default_bits": default_bits,
            "layer_rules": layer_rules or [],
            "output_path": output_path,
        })

    async def get_layered_quantize_job(self, job_id: str) -> dict:
        return await self._get(f"/quantize/layered/jobs/{job_id}")

    async def list_layered_quantize_jobs(self) -> dict:
        return await self._get("/quantize/layered/jobs")

    async def evaluate_quantize(
        self, source_version_id: str, quant_bits: int = 4, sample_size: int = 128,
    ) -> dict:
        return await self._post("/quantize/evaluate", json={
            "source_version_id": source_version_id,
            "quant_bits": quant_bits,
            "sample_size": sample_size,
        })

    # --- Inference ---
    async def chat_completions(self, model: str, messages: list[dict], **kwargs: Any) -> dict:
        return await self._post("/inference/chat/completions", json={
            "model": model, "messages": messages, **kwargs,
        })

    async def completions(self, model: str, prompt: str, **kwargs: Any) -> dict:
        return await self._post("/inference/completions", json={
            "model": model, "prompt": prompt, **kwargs,
        })

    async def embeddings(self, model: str, input: str | list[str], **kwargs: Any) -> dict:
        return await self._post("/inference/embeddings", json={
            "model": model, "input": input, **kwargs,
        })

    # --- Security ---
    async def start_security_scan(self, version_id: str, scan_type: str = "full") -> dict:
        return await self._post("/security/scan", json={
            "version_id": version_id, "scan_type": scan_type,
        })

    async def get_security_scan(self, scan_id: str) -> dict:
        return await self._get(f"/security/scan/{scan_id}")

    async def list_security_scans(self, **params: Any) -> dict:
        return await self._get("/security/scans", params=params)

    # --- Watermark ---
    async def embed_watermark(self, version_id: str, metadata: str = "{}") -> dict:
        return await self._post("/watermark/embed", json={
            "version_id": version_id, "metadata": metadata,
        })

    async def verify_watermark(self, version_id: str) -> dict:
        return await self._post("/watermark/verify", json={"version_id": version_id})

    async def list_watermarks(self, **params: Any) -> dict:
        return await self._get("/watermark/list", params=params)

    # --- Encryption ---
    async def encrypt_version(self, version_id: str) -> dict:
        return await self._post("/encryption/encrypt", json={"version_id": version_id})

    async def decrypt_version(self, version_id: str) -> dict:
        return await self._post("/encryption/decrypt", json={"version_id": version_id})

    async def get_encryption_status(self, version_id: str) -> dict:
        return await self._get(f"/encryption/status/{version_id}")

    # --- Approvals ---
    async def create_approval(self, version_id: str, level: str = "L2", reason: str = "") -> dict:
        return await self._post("/approvals", json={
            "version_id": version_id, "level": level, "reason": reason,
        })

    async def list_approvals(self, **params: Any) -> dict:
        return await self._get("/approvals", params=params)

    async def get_approval(self, req_id: str) -> dict:
        return await self._get(f"/approvals/{req_id}")

    async def approve_request(self, req_id: str, comment: str = "") -> dict:
        return await self._post(f"/approvals/{req_id}/approve", json={"comment": comment})

    async def reject_request(self, req_id: str, comment: str = "") -> dict:
        return await self._post(f"/approvals/{req_id}/reject", json={"comment": comment})

    # --- Git LFS ---
    async def gitlfs_batch(self, operation: str, objects: list[dict]) -> dict:
        return await self._post("/gitlfs/objects/batch", json={
            "operation": operation, "objects": objects,
        })

    async def create_gitlfs_lock(self, path: str) -> dict:
        return await self._post("/gitlfs/locks", json={"path": path})

    async def list_gitlfs_locks(self, **params: Any) -> dict:
        return await self._get("/gitlfs/locks", params=params)

    async def delete_gitlfs_lock(self, lock_id: str) -> dict:
        return await self._delete(f"/gitlfs/locks/{lock_id}")

    # --- Cluster ---
    async def list_nodes(self) -> list[dict]:
        return await self._get("/cluster/nodes")

    async def add_node(self, name: str, url: str, capabilities: str = "inference,quantize") -> dict:
        return await self._post("/cluster/nodes", json={
            "name": name, "url": url, "capabilities": capabilities,
        })

    async def get_node(self, node_id: str) -> dict:
        return await self._get(f"/cluster/nodes/{node_id}")

    async def remove_node(self, node_id: str) -> dict:
        return await self._delete(f"/cluster/nodes/{node_id}")

    async def submit_distributed_task(
        self, task_type: str, model_version_id: str,
        target_node_ids: list[str] | None = None, config: str = "{}",
    ) -> dict:
        body: dict[str, Any] = {
            "task_type": task_type,
            "model_version_id": model_version_id,
            "config": config,
        }
        if target_node_ids:
            body["target_node_ids"] = target_node_ids
        return await self._post("/cluster/distributed-tasks", json=body)

    async def get_distributed_task(self, task_id: str) -> dict:
        return await self._get(f"/cluster/distributed-tasks/{task_id}")

    # --- Ratings ---
    async def create_rating(self, model_id: str, score: int, comment: str = "") -> dict:
        return await self._post(f"/models/{model_id}/ratings", json={
            "score": score, "comment": comment,
        })

    async def list_ratings(self, model_id: str, **params: Any) -> dict:
        return await self._get(f"/models/{model_id}/ratings", params=params)

    async def get_rating_summary(self, model_id: str) -> dict:
        return await self._get(f"/models/{model_id}/ratings/summary")

    async def delete_rating(self, rating_id: str) -> dict:
        return await self._delete(f"/models/ratings/{rating_id}")

    # --- Favorites ---
    async def add_favorite(self, model_id: str) -> dict:
        return await self._post(f"/models/{model_id}/favorites")

    async def list_favorites(self, model_id: str, **params: Any) -> dict:
        return await self._get(f"/models/{model_id}/favorites", params=params)

    async def list_my_favorites(self, **params: Any) -> dict:
        return await self._get("/models/favorites/me", params=params)

    async def remove_favorite(self, favorite_id: str) -> dict:
        return await self._delete(f"/models/favorites/{favorite_id}")

    # --- Branches ---
    async def create_branch(self, model_id: str, name: str, base_version_id: str = "", description: str = "") -> dict:
        return await self._post(f"/models/{model_id}/branches", json={
            "name": name, "base_version_id": base_version_id, "description": description,
        })

    async def list_branches(self, model_id: str, status: str = "") -> dict:
        return await self._get(f"/models/{model_id}/branches", params={"status": status} if status else {})

    async def get_branch(self, branch_id: str) -> dict:
        return await self._get(f"/models/branches/{branch_id}")

    async def update_branch(self, branch_id: str, data: dict) -> dict:
        return await self._put(f"/models/branches/{branch_id}", json=data)

    async def delete_branch(self, branch_id: str) -> dict:
        return await self._delete(f"/models/branches/{branch_id}")

    async def merge_branch(self, branch_id: str) -> dict:
        return await self._post(f"/models/branches/{branch_id}/merge")

    # --- Hardware ---
    async def get_hardware_info(self) -> dict:
        return await self._get("/hardware")

    async def refresh_hardware(self) -> dict:
        return await self._post("/hardware/refresh")

    # --- Recommend ---
    async def recommend_models(
        self, task: str = "llm", preference: str = "balanced",
        max_results: int = 10, min_params: float = 0, max_params: float = 1000,
    ) -> dict:
        return await self._post("/recommend", json={
            "task": task, "preference": preference,
            "max_results": max_results, "min_params_b": min_params, "max_params_b": max_params,
        })

    async def quick_recommend(self, task: str = "llm", preference: str = "balanced") -> dict:
        return await self._get("/recommend/quick", params={"task": task, "preference": preference})

    # --- Adapt ---
    async def assess_model(
        self, model_id: str, hf_repo: str | None = None,
        source_format: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"model_id": model_id}
        if hf_repo:
            body["hf_repo"] = hf_repo
        if source_format:
            body["source_format"] = source_format
        return await self._post("/adapt/assess", json=body)

    async def plan_migration(
        self, model_id: str, params_b: float = 0,
        hf_repo: str | None = None, source_format: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"model_id": model_id, "params_b": params_b}
        if hf_repo:
            body["hf_repo"] = hf_repo
        if source_format:
            body["source_format"] = source_format
        return await self._post("/adapt/plan", json=body)

    async def execute_adaptation(
        self, model_id: str, hf_repo: str | None = None,
        source_format: str | None = None, quant_bits: int = 4,
        params_b: float = 0,
    ) -> dict:
        body: dict[str, Any] = {
            "model_id": model_id, "quant_bits": quant_bits, "params_b": params_b,
        }
        if hf_repo:
            body["hf_repo"] = hf_repo
        if source_format:
            body["source_format"] = source_format
        return await self._post("/adapt/execute", json=body)

    async def get_adapt_execution(self, execution_id: str) -> dict:
        return await self._get(f"/adapt/execute/{execution_id}")

    # --- Benchmarks ---
    async def list_benchmarks(
        self, chip: str | None = None, model_id: str | None = None,
        quant: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {}
        if chip:
            params["chip"] = chip
        if model_id:
            params["model_id"] = model_id
        if quant:
            params["quant"] = quant
        return await self._get("/benchmarks", params=params)

    async def get_benchmark(
        self, model_id: str, chip: str | None = None, quant: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {}
        if chip:
            params["chip"] = chip
        if quant:
            params["quant"] = quant
        return await self._get(f"/benchmarks/{model_id}", params=params)

    # --- Analyze ---
    async def analyze_model(
        self, model_path: str | None = None, hf_repo: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if model_path:
            body["model_path"] = model_path
        if hf_repo:
            body["hf_repo"] = hf_repo
        return await self._post("/analyze", json=body)

    # --- System ---
    async def health(self) -> dict:
        return await self._get("/system/health")

    async def storage_stats(self) -> dict:
        return await self._get("/system/storage")

    async def export_data(self, **params: Any) -> dict:
        return await self._get("/system/export", params=params)

    # --- Auth ---
    async def create_api_key(self, name: str) -> dict:
        return await self._post("/auth/keys", json={"name": name})

    async def list_api_keys(self) -> dict:
        return await self._get("/auth/keys")

    async def deactivate_api_key(self, key_id: str) -> dict:
        return await self._post(f"/auth/keys/{key_id}/deactivate")

    async def delete_api_key(self, key_id: str) -> dict:
        return await self._delete(f"/auth/keys/{key_id}")
