import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class FusionModelHubClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11444",
        api_key: str | None = None,
        timeout: float = 30.0,
        *,
        verify: bool | str = True,
        cert: str | tuple[str, str] | tuple[str, str, str] | None = None,
        trust_env: bool = True,
    ):
        # E-E14: prior __init__ accepted no TLS controls and the default base_url
        # is http:// (plaintext) — fine for local loopback, but a user pointing
        # the SDK at a remote Hub over https had no way to configure client cert
        # auth or a custom CA bundle, and could only disable verification by
        # reaching into httpx globally. Expose verify/cert/trust_env and thread
        # them into a single persistent httpx.Client so connections are reused
        # (the prior code opened a new Client per request = pool churn).
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._timeout = timeout
        self._verify = verify
        self._cert = cert
        self._trust_env = trust_env
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=self._timeout,
                headers=self._headers,
                verify=self._verify,
                cert=self._cert,
                trust_env=self._trust_env,
            )
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _url(self, path: str) -> str:
        return f"{self._base_url}/api/v1{path}"

    def _get(self, path: str, params: dict | None = None) -> dict:
        c = self._get_client()
        r = c.get(self._url(path), params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict | None = None, **kwargs: Any) -> dict:
        c = self._get_client()
        r = c.post(self._url(path), json=json, **kwargs)
        r.raise_for_status()
        return r.json()

    def _put(self, path: str, json: dict | None = None) -> dict:
        c = self._get_client()
        r = c.put(self._url(path), json=json)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, params: dict | None = None) -> dict:
        c = self._get_client()
        r = c.delete(self._url(path), params=params)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, json: dict | None = None) -> dict:
        c = self._get_client()
        r = c.patch(self._url(path), json=json)
        r.raise_for_status()
        return r.json()

    # --- Models ---
    def list_models(self, **params: Any) -> dict:
        return self._get("/models", params=params)

    def get_model(self, model_id: str) -> dict:
        return self._get(f"/models/{model_id}")

    def create_model(self, data: dict) -> dict:
        return self._post("/models", json=data)

    def update_model(self, model_id: str, data: dict) -> dict:
        return self._put(f"/models/{model_id}", json=data)

    def delete_model(self, model_id: str) -> dict:
        return self._delete(f"/models/{model_id}")

    def import_from_hf(self, data: dict) -> dict:
        return self._post("/models/import/hf", json=data)

    def sync_registry(self, source_url: str) -> dict:
        return self._post("/models/sync", json={"source_url": source_url})

    def batch_delete(self, model_ids: list[str]) -> dict:
        return self._post("/models/batch-delete", json={"model_ids": model_ids})

    def compare_models(self, model_ids: list[str]) -> dict:
        return self._get("/models/compare", params={"model_ids": ",".join(model_ids)})

    # --- Versions ---
    def list_versions(self, model_id: str, **params: Any) -> dict:
        return self._get(f"/models/{model_id}/versions", params=params)

    def get_version(self, version_id: str) -> dict:
        return self._get(f"/versions/{version_id}")

    def update_version(self, version_id: str, data: dict) -> dict:
        return self._put(f"/versions/{version_id}", json=data)

    def delete_version(self, version_id: str) -> dict:
        return self._delete(f"/versions/{version_id}")

    def promote_version(self, version_id: str) -> dict:
        return self._post(f"/versions/{version_id}/promote")

    def benchmark_version(self, version_id: str) -> dict:
        return self._post(f"/versions/{version_id}/benchmark")

    def rollback_version(self, version_id: str) -> dict:
        return self._post(f"/versions/{version_id}/rollback")

    def deprecate_version(self, version_id: str) -> dict:
        return self._post(f"/versions/{version_id}/deprecate")

    def retire_version(self, version_id: str) -> dict:
        return self._post(f"/versions/{version_id}/retire")

    # --- Quantize ---
    def start_quantize(
        self,
        source_version_id: str,
        target_format: str = "mlx",
        quant_bits: int = 4,
        calibration_dataset: str = "",
    ) -> dict:
        return self._post(
            "/quantize",
            json={
                "source_version_id": source_version_id,
                "target_format": target_format,
                "quant_bits": quant_bits,
                "calibration_dataset": calibration_dataset,
            },
        )

    def list_quantize_tasks(self, **params: Any) -> dict:
        return self._get("/quantize", params=params)

    def get_quantize_status(self, task_id: str) -> dict:
        return self._get(f"/quantize/{task_id}")

    def start_lora_merge(
        self,
        base_version_id: str,
        lora_version_id: str,
        target_format: str = "mlx",
        quant_bits: int = 4,
    ) -> dict:
        return self._post(
            "/quantize/lora-merge",
            json={
                "base_version_id": base_version_id,
                "lora_version_id": lora_version_id,
                "target_format": target_format,
                "quant_bits": quant_bits,
            },
        )

    def get_lora_merge_status(self, task_id: str) -> dict:
        return self._get(f"/quantize/lora-merge/{task_id}")

    def start_layered_quantize(
        self,
        model: str,
        default_bits: int = 4,
        layer_rules: list[dict] | None = None,
        output_path: str = "",
    ) -> dict:
        return self._post(
            "/quantize/layered",
            json={
                "model": model,
                "default_bits": default_bits,
                "layer_rules": layer_rules or [],
                "output_path": output_path,
            },
        )

    def get_layered_quantize_job(self, job_id: str) -> dict:
        return self._get(f"/quantize/layered/jobs/{job_id}")

    def list_layered_quantize_jobs(self) -> dict:
        return self._get("/quantize/layered/jobs")

    def evaluate_quantize(
        self,
        source_version_id: str,
        quant_bits: int = 4,
        sample_size: int = 128,
    ) -> dict:
        return self._post(
            "/quantize/evaluate",
            json={
                "source_version_id": source_version_id,
                "quant_bits": quant_bits,
                "sample_size": sample_size,
            },
        )

    # --- Inference ---
    def chat_completions(self, model: str, messages: list[dict], **kwargs: Any) -> dict:
        return self._post(
            "/inference/chat/completions",
            json={
                "model": model,
                "messages": messages,
                **kwargs,
            },
        )

    def completions(self, model: str, prompt: str, **kwargs: Any) -> dict:
        return self._post(
            "/inference/completions",
            json={
                "model": model,
                "prompt": prompt,
                **kwargs,
            },
        )

    def embeddings(self, model: str, input: str | list[str], **kwargs: Any) -> dict:
        return self._post(
            "/inference/embeddings",
            json={
                "model": model,
                "input": input,
                **kwargs,
            },
        )

    # --- Security ---
    def start_security_scan(self, version_id: str, scan_type: str = "full") -> dict:
        return self._post(
            "/security/scan",
            json={
                "version_id": version_id,
                "scan_type": scan_type,
            },
        )

    def get_security_scan(self, scan_id: str) -> dict:
        return self._get(f"/security/scan/{scan_id}")

    def list_security_scans(self, **params: Any) -> dict:
        return self._get("/security/scans", params=params)

    # --- Watermark ---
    def embed_watermark(self, version_id: str, metadata: str = "{}") -> dict:
        return self._post(
            "/watermark/embed",
            json={
                "version_id": version_id,
                "metadata": metadata,
            },
        )

    def verify_watermark(self, version_id: str) -> dict:
        return self._post("/watermark/verify", json={"version_id": version_id})

    def list_watermarks(self, **params: Any) -> dict:
        return self._get("/watermark/list", params=params)

    # --- Encryption ---
    def encrypt_version(self, version_id: str) -> dict:
        return self._post("/encryption/encrypt", json={"version_id": version_id})

    def decrypt_version(self, version_id: str) -> dict:
        return self._post("/encryption/decrypt", json={"version_id": version_id})

    def get_encryption_status(self, version_id: str) -> dict:
        return self._get(f"/encryption/status/{version_id}")

    # --- Approvals ---
    def create_approval(self, version_id: str, level: str = "L2", reason: str = "") -> dict:
        return self._post(
            "/approvals",
            json={
                "version_id": version_id,
                "level": level,
                "reason": reason,
            },
        )

    def list_approvals(self, **params: Any) -> dict:
        return self._get("/approvals", params=params)

    def get_approval(self, req_id: str) -> dict:
        return self._get(f"/approvals/{req_id}")

    def approve_request(self, req_id: str, comment: str = "") -> dict:
        return self._post(f"/approvals/{req_id}/approve", json={"comment": comment})

    def reject_request(self, req_id: str, comment: str = "") -> dict:
        return self._post(f"/approvals/{req_id}/reject", json={"comment": comment})

    # --- Git LFS ---
    def gitlfs_batch(self, operation: str, objects: list[dict]) -> dict:
        return self._post(
            "/gitlfs/objects/batch",
            json={
                "operation": operation,
                "objects": objects,
            },
        )

    def create_gitlfs_lock(self, path: str) -> dict:
        return self._post("/gitlfs/locks", json={"path": path})

    def list_gitlfs_locks(self, **params: Any) -> dict:
        return self._get("/gitlfs/locks", params=params)

    def delete_gitlfs_lock(self, lock_id: str) -> dict:
        return self._delete(f"/gitlfs/locks/{lock_id}")

    # --- Cluster ---
    def list_nodes(self) -> list[dict]:
        return self._get("/cluster/nodes")

    def add_node(self, name: str, url: str, capabilities: str = "inference,quantize") -> dict:
        return self._post(
            "/cluster/nodes",
            json={
                "name": name,
                "url": url,
                "capabilities": capabilities,
            },
        )

    def get_node(self, node_id: str) -> dict:
        return self._get(f"/cluster/nodes/{node_id}")

    def remove_node(self, node_id: str) -> dict:
        return self._delete(f"/cluster/nodes/{node_id}")

    def submit_distributed_task(
        self,
        task_type: str,
        model_version_id: str,
        target_node_ids: list[str] | None = None,
        config: str = "{}",
    ) -> dict:
        body: dict[str, Any] = {
            "task_type": task_type,
            "model_version_id": model_version_id,
            "config": config,
        }
        if target_node_ids:
            body["target_node_ids"] = target_node_ids
        return self._post("/cluster/distributed-tasks", json=body)

    def get_distributed_task(self, task_id: str) -> dict:
        return self._get(f"/cluster/distributed-tasks/{task_id}")

    # --- Hardware ---
    def get_hardware_info(self) -> dict:
        return self._get("/hardware")

    def refresh_hardware(self) -> dict:
        return self._post("/hardware/refresh")

    # --- Recommend ---
    def recommend_models(
        self,
        task: str = "llm",
        preference: str = "balanced",
        max_results: int = 10,
        min_params: float = 0,
        max_params: float = 1000,
    ) -> dict:
        return self._post(
            "/recommend",
            json={
                "task": task,
                "preference": preference,
                "max_results": max_results,
                "min_params_b": min_params,
                "max_params_b": max_params,
            },
        )

    def quick_recommend(self, task: str = "llm", preference: str = "balanced") -> dict:
        return self._get("/recommend/quick", params={"task": task, "preference": preference})

    # --- Adapt ---
    def assess_model(
        self,
        model_id: str,
        hf_repo: str | None = None,
        source_format: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"model_id": model_id}
        if hf_repo:
            body["hf_repo"] = hf_repo
        if source_format:
            body["source_format"] = source_format
        return self._post("/adapt/assess", json=body)

    def plan_migration(
        self,
        model_id: str,
        params_b: float = 0,
        hf_repo: str | None = None,
        source_format: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"model_id": model_id, "params_b": params_b}
        if hf_repo:
            body["hf_repo"] = hf_repo
        if source_format:
            body["source_format"] = source_format
        return self._post("/adapt/plan", json=body)

    def execute_adaptation(
        self,
        model_id: str,
        hf_repo: str | None = None,
        source_format: str | None = None,
        quant_bits: int = 4,
        params_b: float = 0,
    ) -> dict:
        body: dict[str, Any] = {
            "model_id": model_id,
            "quant_bits": quant_bits,
            "params_b": params_b,
        }
        if hf_repo:
            body["hf_repo"] = hf_repo
        if source_format:
            body["source_format"] = source_format
        return self._post("/adapt/execute", json=body)

    def get_adapt_execution(self, execution_id: str) -> dict:
        return self._get(f"/adapt/execute/{execution_id}")

    # --- Benchmarks ---
    def list_benchmarks(
        self,
        chip: str | None = None,
        model_id: str | None = None,
        quant: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {}
        if chip:
            params["chip"] = chip
        if model_id:
            params["model_id"] = model_id
        if quant:
            params["quant"] = quant
        return self._get("/benchmarks", params=params)

    def get_benchmark(
        self,
        model_id: str,
        chip: str | None = None,
        quant: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {}
        if chip:
            params["chip"] = chip
        if quant:
            params["quant"] = quant
        return self._get(f"/benchmarks/{model_id}", params=params)

    # --- Analyze ---
    def analyze_model(
        self,
        model_path: str | None = None,
        hf_repo: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if model_path:
            body["model_path"] = model_path
        if hf_repo:
            body["hf_repo"] = hf_repo
        return self._post("/analyze", json=body)

    # --- System ---
    def health(self) -> dict:
        return self._get("/system/health")

    def storage_stats(self) -> dict:
        return self._get("/system/storage")

    def export_data(self, **params: Any) -> dict:
        return self._get("/system/export", params=params)

    # --- Auth ---
    def create_api_key(self, name: str) -> dict:
        return self._post("/auth/keys", json={"name": name})

    def list_api_keys(self) -> dict:
        return self._get("/auth/keys")

    def deactivate_api_key(self, key_id: str) -> dict:
        return self._post(f"/auth/keys/{key_id}/deactivate")

    def delete_api_key(self, key_id: str) -> dict:
        return self._delete(f"/auth/keys/{key_id}")

    # --- Ratings ---
    def create_rating(self, model_id: str, score: int, comment: str = "") -> dict:
        return self._post(
            f"/models/{model_id}/ratings",
            json={
                "score": score,
                "comment": comment,
            },
        )

    def list_ratings(self, model_id: str, **params: Any) -> dict:
        return self._get(f"/models/{model_id}/ratings", params=params)

    def get_rating_summary(self, model_id: str) -> dict:
        return self._get(f"/models/{model_id}/ratings/summary")

    def delete_rating(self, rating_id: str) -> dict:
        return self._delete(f"/models/ratings/{rating_id}")

    # --- Favorites ---
    def add_favorite(self, model_id: str) -> dict:
        return self._post(f"/models/{model_id}/favorites")

    def list_favorites(self, model_id: str, **params: Any) -> dict:
        return self._get(f"/models/{model_id}/favorites", params=params)

    def list_my_favorites(self, **params: Any) -> dict:
        return self._get("/models/favorites/me", params=params)

    def remove_favorite(self, favorite_id: str) -> dict:
        return self._delete(f"/models/favorites/{favorite_id}")

    # --- Branches ---
    def create_branch(self, model_id: str, name: str, base_version_id: str = "", description: str = "") -> dict:
        return self._post(
            f"/models/{model_id}/branches",
            json={
                "name": name,
                "base_version_id": base_version_id,
                "description": description,
            },
        )

    def list_branches(self, model_id: str, status: str = "") -> dict:
        return self._get(f"/models/{model_id}/branches", params={"status": status} if status else {})

    def get_branch(self, branch_id: str) -> dict:
        return self._get(f"/models/branches/{branch_id}")

    def update_branch(self, branch_id: str, data: dict) -> dict:
        return self._put(f"/models/branches/{branch_id}", json=data)

    def delete_branch(self, branch_id: str) -> dict:
        return self._delete(f"/models/branches/{branch_id}")

    def merge_branch(self, branch_id: str) -> dict:
        return self._post(f"/models/branches/{branch_id}/merge")

    # --- Model serve lifecycle ---
    def publish_model(self, model_id: str) -> dict:
        return self._post(f"/models/{model_id}/publish")

    def serve_model(self, model_id: str, version_id: str = "", gpu: bool = True) -> dict:
        return self._post(f"/models/{model_id}/serve", json={"version_id": version_id, "gpu": gpu})

    def unload_model(self, model_id: str) -> dict:
        return self._delete(f"/models/{model_id}/serve")

    def serve_status(self, model_id: str) -> dict:
        return self._get(f"/models/{model_id}/serve")

    def hot_reload_model(self, model_id: str, version_id: str) -> dict:
        return self._post(f"/models/{model_id}/hot-reload", json={"version_id": version_id})

    # --- Cache (3-level: raw → converted → quantized) ---
    def cache_stats(self) -> dict:
        return self._get("/cache")

    def cache_list_entries(self, level: str = "") -> dict:
        return self._get("/cache/entries", params={"level": level} if level else {})

    def cache_gc(self, max_size_gb: float = 0, max_age_days: float = 30) -> dict:
        return self._post("/cache/gc", params={"max_size_gb": max_size_gb, "max_age_days": max_age_days})

    def cache_validate(self, mlx_version: str = "") -> dict:
        return self._post("/cache/validate", params={"mlx_version": mlx_version})

    def cache_remove_model(self, model_id: str) -> dict:
        return self._delete(f"/cache/{model_id}")

    def cache_remove_entry(self, model_id: str, level: str, quant_bits: int = 0) -> dict:
        return self._delete(f"/cache/{model_id}/{level}", params={"quant_bits": quant_bits} if quant_bits else {})

    # --- Deployments ---
    def list_deployments(self, model_id: str = "", status: str = "", page: int = 1, page_size: int = 20) -> dict:
        return self._get(
            "/deployments", params={"model_id": model_id, "status": status, "page": page, "page_size": page_size}
        )

    def create_deployment(self, data: dict) -> dict:
        return self._post("/deployments", json=data)

    def get_deployment(self, deployment_id: str) -> dict:
        return self._get(f"/deployments/{deployment_id}")

    def update_deployment(self, deployment_id: str, data: dict) -> dict:
        return self._patch(f"/deployments/{deployment_id}", json=data)

    def delete_deployment(self, deployment_id: str) -> dict:
        return self._delete(f"/deployments/{deployment_id}")

    def stop_deployment(self, deployment_id: str) -> dict:
        return self._post(f"/deployments/{deployment_id}/stop")

    def gray_release(self, deployment_id: str, data: dict | None = None) -> dict:
        return self._post(f"/deployments/{deployment_id}/gray", json=data or {})

    def stop_gray_release(self, deployment_id: str) -> dict:
        return self._delete(f"/deployments/{deployment_id}/gray")

    def scale_deployment(self, deployment_id: str, scale: int) -> dict:
        return self._post(f"/deployments/{deployment_id}/scale", json={"scale": scale})

    def deployment_metrics(self, deployment_id: str) -> dict:
        return self._get(f"/deployments/{deployment_id}/metrics")

    # --- Downloads ---
    def create_download(
        self,
        model_id: str,
        source_url: str,
        version_id: str = "",
        speed_limit_kbps: int = 0,
        max_retries: int = 3,
        expected_sha256: str = "",
    ) -> dict:
        return self._post(
            "/downloads",
            json={
                "model_id": model_id,
                "source_url": source_url,
                "version_id": version_id,
                "speed_limit_kbps": speed_limit_kbps,
                "max_retries": max_retries,
                "expected_sha256": expected_sha256,
            },
        )

    def list_downloads(self, model_id: str = "", status: str = "", page: int = 1, page_size: int = 20) -> dict:
        return self._get(
            "/downloads", params={"model_id": model_id, "status": status, "page": page, "page_size": page_size}
        )

    def get_download(self, task_id: str) -> dict:
        return self._get(f"/downloads/{task_id}")

    def cancel_download(self, task_id: str) -> dict:
        return self._delete(f"/downloads/{task_id}")

    # --- Evaluations ---
    def list_evaluations(
        self,
        model_id: str = "",
        version_id: str = "",
        benchmark_name: str = "",
        status: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        return self._get(
            "/evaluations",
            params={
                "model_id": model_id,
                "version_id": version_id,
                "benchmark_name": benchmark_name,
                "status": status,
                "page": page,
                "page_size": page_size,
            },
        )

    def create_evaluation(self, model_id: str, benchmark_name: str, version_id: str = "") -> dict:
        return self._post(
            "/evaluations", json={"model_id": model_id, "benchmark_name": benchmark_name, "version_id": version_id}
        )

    def compare_benchmarks(self, model_id: str = "", benchmark_name: str = "") -> dict:
        return self._get(
            "/evaluations/benchmarks/compare", params={"model_id": model_id, "benchmark_name": benchmark_name}
        )

    def get_evaluation(self, eval_id: str) -> dict:
        return self._get(f"/evaluations/{eval_id}")

    def update_evaluation(self, eval_id: str, data: dict) -> dict:
        return self._patch(f"/evaluations/{eval_id}", json=data)

    def delete_evaluation(self, eval_id: str) -> dict:
        return self._delete(f"/evaluations/{eval_id}")

    # --- Tenants + roles ---
    def list_tenants(self) -> dict:
        return self._get("/tenants")

    def create_tenant(self, name: str, display_name: str = "") -> dict:
        return self._post("/tenants", json={"name": name, "display_name": display_name})

    def get_tenant(self, tenant_id: str) -> dict:
        return self._get(f"/tenants/{tenant_id}")

    def update_tenant(self, tenant_id: str, display_name: str = "") -> dict:
        return self._patch(f"/tenants/{tenant_id}", json={"display_name": display_name})

    def delete_tenant(self, tenant_id: str) -> dict:
        return self._delete(f"/tenants/{tenant_id}")

    def list_roles(self, tenant_id: str) -> dict:
        return self._get(f"/tenants/{tenant_id}/roles")

    def create_role(self, tenant_id: str, name: str, permissions: str = "read") -> dict:
        return self._post(f"/tenants/{tenant_id}/roles", json={"name": name, "permissions": permissions})

    def update_role(self, tenant_id: str, role_id: str, data: dict) -> dict:
        return self._put(f"/tenants/{tenant_id}/roles/{role_id}", json=data)

    def delete_role(self, tenant_id: str, role_id: str) -> dict:
        return self._delete(f"/tenants/{tenant_id}/roles/{role_id}")

    # --- Webhooks ---
    def list_webhooks(self) -> dict:
        return self._get("/webhooks")

    def create_webhook(
        self, name: str, url: str, secret: str = "", events: str = "model.created,model.deleted"
    ) -> dict:
        return self._post("/webhooks", json={"name": name, "url": url, "secret": secret, "events": events})

    def get_webhook(self, webhook_id: str) -> dict:
        return self._get(f"/webhooks/{webhook_id}")

    def delete_webhook(self, webhook_id: str) -> dict:
        return self._delete(f"/webhooks/{webhook_id}")

    # --- Monitor ---
    def realtime_monitor(self) -> dict:
        return self._get("/monitor/realtime")

    def model_stats(self) -> dict:
        return self._get("/monitor/model-stats")
