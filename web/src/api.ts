// issue #56: fusion-model-hub REST API client (:11444). Same-origin fetch —
// standalone dev goes through the vite proxy, embedded through the gateway
// /shell/fusion-models-hub/ proxy.
//
// Auth: the hub requires X-API-Key on write endpoints (POST/PUT/DELETE) when
// MODEL_HUB_AUTH_ENABLED=true. The UI has NO auth logic (spec S8) — embedded
// mode the shell/gateway session injects the key upstream; standalone dev
// wires it from VITE_HUB_API_KEY (dev convenience only). Reads work
// anonymously.
const API_KEY: string | undefined = import.meta.env.VITE_HUB_API_KEY;

export const HF_MIRROR = "https://hf-mirror.com";

export interface ModelTag {
    key: string;
    value: string;
}

export interface ModelSummary {
    id: string;
    name: string;
    description: string;
    model_type: string;
    base_model_id: string;
    architecture: string;
    params_size: string;
    license: string;
    author: string;
    hf_repo: string;
    download_count: number;
    model_status: string;
    tags: ModelTag[];
    versions_count: number;
    created_at: string | null;
    updated_at: string | null;
}

export interface ModelVersion {
    id: string;
    version: string;
    format: string;
    quantization: string;
    status: string;
    file_size: number;
    created_at: string | null;
}

export interface ModelDetail extends ModelSummary {
    versions: ModelVersion[];
}

export interface DownloadTask {
    task_id: string;
    model_id: string;
    version_id: string;
    source_url: string;
    status: string;
    progress_percent: number;
    downloaded_bytes: number;
    total_bytes: number;
    retry_count: number;
    max_retries: number;
    error_message: string | null;
    created_at: string | null;
}

export interface QuantizeTask {
    id: string;
    source_version_id: string;
    target_format: string;
    quant_bits: number;
    calibration_dataset: string;
    status: string;
    output_version_id: string;
    error_message: string | null;
    started_at: string | null;
    completed_at: string | null;
    created_at: string | null;
}

export interface StorageStats {
    path?: string;
    endpoint?: string;
    bucket?: string;
    model_count: number;
    file_count: number;
    total_size_gb: number;
}

export interface HealthInfo {
    status: string;
    version?: string;
    mlx?: { status: string; url?: string };
    storage?: Record<string, unknown>;
    data_dir?: string;
    [key: string]: unknown;
}

function headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (API_KEY) h["X-API-Key"] = API_KEY;
    return h;
}

async function readError(resp: Response): Promise<string> {
    const body = (await resp.json().catch(() => null)) as {
        detail?: string;
        message?: string;
    } | null;
    return body?.detail ?? body?.message ?? `HTTP ${resp.status}`;
}

async function getJson<T>(path: string): Promise<T> {
    const resp = await fetch(path, { headers: headers() });
    if (!resp.ok) {
        throw new Error(await readError(resp));
    }
    return (await resp.json()) as T;
}

async function sendJson<T>(
    method: "POST" | "DELETE",
    path: string,
    body?: unknown,
): Promise<T> {
    const resp = await fetch(path, {
        method,
        headers: headers(),
        body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!resp.ok) {
        throw new Error(await readError(resp));
    }
    return (await resp.json().catch(() => ({}) as T)) as T;
}

export async function fetchModels(
    keyword = "",
    page = 1,
    pageSize = 20,
): Promise<ModelSummary[]> {
    const data = await getJson<{ items: ModelSummary[] }>(
        `/api/v1/models?keyword=${encodeURIComponent(keyword)}&page=${page}&page_size=${pageSize}`,
    );
    return data.items ?? [];
}

export async function fetchModel(modelId: string): Promise<ModelDetail> {
    return getJson<ModelDetail>(
        `/api/v1/models/${encodeURIComponent(modelId)}`,
    );
}

export async function importFromHf(
    hfRepo: string,
    name = "",
): Promise<ModelSummary> {
    return sendJson<ModelSummary>("POST", "/api/v1/models/import/hf", {
        hf_repo: hfRepo,
        name,
        download: false,
    });
}

export async function fetchDownloads(
    page = 1,
    pageSize = 50,
): Promise<DownloadTask[]> {
    const data = await getJson<{ tasks: DownloadTask[] }>(
        `/api/v1/downloads?page=${page}&page_size=${pageSize}`,
    );
    return data.tasks ?? [];
}

export async function createDownload(
    modelId: string,
    sourceUrl: string,
): Promise<{ task_id: string }> {
    return sendJson<{ task_id: string }>("POST", "/api/v1/downloads", {
        model_id: modelId,
        source_url: sourceUrl,
    });
}

export async function cancelDownload(taskId: string): Promise<void> {
    await sendJson("DELETE", `/api/v1/downloads/${encodeURIComponent(taskId)}`);
}

export async function fetchQuantizeTasks(
    status = "",
    page = 1,
    pageSize = 20,
): Promise<QuantizeTask[]> {
    const data = await getJson<{ items: QuantizeTask[] }>(
        `/api/v1/quantize?status=${encodeURIComponent(status)}&page=${page}&page_size=${pageSize}`,
    );
    return data.items ?? [];
}

export async function submitQuantize(input: {
    source_version_id: string;
    target_format: string;
    quant_bits: number;
    calibration_dataset?: string;
}): Promise<{ task_id: string }> {
    return sendJson<{ task_id: string }>("POST", "/api/v1/quantize", input);
}

export async function fetchStorage(): Promise<StorageStats> {
    return getJson<StorageStats>("/api/v1/system/storage");
}

export async function fetchHealth(): Promise<HealthInfo> {
    return getJson<HealthInfo>("/api/v1/system/health");
}
