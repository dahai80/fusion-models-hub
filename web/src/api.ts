// issue #56: fusion-model-hub REST API client (:11444). Same-origin fetch —
// standalone dev goes through the vite proxy, embedded through the gateway
// /shell/fusion-models-hub/ proxy.
//
// issue #88: API calls must ride the SAME prefix the iframe is mounted under.
// The vite `base: "./"` only fixes static assets — a fetch("/api/v1/...") from
// inside the shell iframe resolves against the GATEWAY origin (:11432), hits
// the gateway's own /api/v1 surface (no session-authed /api/v1/models route)
// and 401s, which the TabView crash card renders as "HTTP 401". Derive the
// prefix from location.pathname: /shell/<module>/... → /shell/<module>,
// standalone ("/" or "/index.html") → "" so same-origin :11444 paths keep
// working unchanged.
export function apiBase(
    pathname = globalThis.location?.pathname ?? "/",
): string {
    const marker = "/shell/";
    const idx = pathname.indexOf(marker);
    if (idx < 0) return "";
    const moduleId = pathname.slice(idx + marker.length).split("/")[0];
    return moduleId ? `${marker}${moduleId}` : "";
}

const API_BASE = apiBase();
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
    const resp = await fetch(`${API_BASE}${path}`, { headers: headers() });
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
    const resp = await fetch(`${API_BASE}${path}`, {
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

// ── issue #90: governance / serving / cluster / versions pages ──

export interface Tenant {
    id: string;
    name: string;
    display_name: string;
    is_active: boolean;
}

export interface ApiKeyInfo {
    id: string;
    name: string;
    tenant_id: string;
    key_prefix: string;
    permissions: string;
    role: string;
    qps_limit: number;
    allowed_models: string;
    allowed_modules: string;
    is_active: boolean;
    created_at: string | null;
    last_used_at: string | null;
}

export interface ApprovalRequest {
    id: string;
    model_id: string;
    version_id: string;
    level: string;
    status: string;
    requester: string;
    approver?: string;
    comment?: string;
    created_at?: string | null;
    [key: string]: unknown;
}

export interface Webhook {
    id: string;
    name: string;
    url: string;
    events: string;
    is_active: boolean;
}

export interface Deployment {
    id: string;
    model_id: string;
    version_id: string;
    name: string;
    replicas: number;
    node_id: string;
    status: string;
    gray_enabled: boolean;
    gray_version_id: string;
    gray_traffic_ratio: number;
}

export interface ClusterNode {
    id: string;
    name: string;
    url: string;
    host: string;
    port: number;
    status: string;
    capabilities: string;
    last_seen: string | null;
}

export interface VersionRecord {
    id: string;
    model_id: string;
    version: string;
    format: string;
    quantization: string;
    status: string;
    file_size: number;
    benchmark_score: number | null;
    created_at?: string | null;
    [key: string]: unknown;
}

export interface SecurityScan {
    id: string;
    model_id: string;
    version_id: string;
    scan_type: string;
    status: string;
    findings: Record<string, unknown>;
    risk_level: string;
    created_at: string | null;
    completed_at: string | null;
}

export interface AuditLogEntry {
    id: string;
    action: string;
    resource_type: string;
    resource_id: string;
    api_key_id: string | null;
    created_at?: string | null;
    [key: string]: unknown;
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
    return sendJson<T>("POST", path, body ?? {});
}

export async function fetchTenants(): Promise<Tenant[]> {
    const data = await getJson<{ tenants: Tenant[] }>("/api/v1/tenants");
    return data.tenants ?? [];
}

export async function createTenant(
    name: string,
    displayName = "",
): Promise<Tenant> {
    return sendJson<Tenant>("POST", "/api/v1/tenants", {
        name,
        display_name: displayName,
    });
}

export async function fetchApiKeys(): Promise<ApiKeyInfo[]> {
    const data = await getJson<{ items: ApiKeyInfo[] }>("/api/v1/auth/keys");
    return data.items ?? [];
}

export async function createApiKey(input: {
    name: string;
    tenant_id?: string;
    role?: string;
}): Promise<{ id: string; key: string; key_prefix: string }> {
    return postJson("/api/v1/auth/keys", input);
}

export async function deactivateApiKey(keyId: string): Promise<void> {
    await postJson(`/api/v1/auth/keys/${encodeURIComponent(keyId)}/deactivate`);
}

export async function fetchApprovals(status = ""): Promise<ApprovalRequest[]> {
    const data = await getJson<{ items: ApprovalRequest[] }>(
        `/api/v1/approvals?status=${encodeURIComponent(status)}&page=1&page_size=50`,
    );
    return data.items ?? [];
}

export async function decideApproval(
    reqId: string,
    action: "approve" | "reject",
    approver = "",
    comment = "",
): Promise<unknown> {
    return postJson(
        `/api/v1/approvals/${encodeURIComponent(reqId)}/${action}`,
        { approver, comment },
    );
}

export async function fetchWebhooks(): Promise<Webhook[]> {
    const data = await getJson<{ webhooks: Webhook[] }>("/api/v1/webhooks");
    return data.webhooks ?? [];
}

export async function createWebhook(input: {
    name: string;
    url: string;
    events?: string;
}): Promise<Webhook> {
    return sendJson<Webhook>("POST", "/api/v1/webhooks", input);
}

export async function deleteWebhook(webhookId: string): Promise<void> {
    await sendJson(
        "DELETE",
        `/api/v1/webhooks/${encodeURIComponent(webhookId)}`,
    );
}

export async function fetchDeployments(): Promise<Deployment[]> {
    const data = await getJson<{ deployments: Deployment[] }>(
        "/api/v1/deployments",
    );
    return data.deployments ?? [];
}

export async function createDeployment(input: {
    model_id: string;
    name?: string;
    replicas?: number;
}): Promise<Deployment> {
    return postJson("/api/v1/deployments", input);
}

export async function scaleDeployment(
    deploymentId: string,
    replicas: number,
): Promise<unknown> {
    return postJson(
        `/api/v1/deployments/${encodeURIComponent(deploymentId)}/scale`,
        { replicas },
    );
}

export async function stopDeployment(deploymentId: string): Promise<unknown> {
    return postJson(
        `/api/v1/deployments/${encodeURIComponent(deploymentId)}/stop`,
    );
}

export async function setGrayRelease(
    deploymentId: string,
    grayVersionId: string,
    grayTrafficRatio = 10,
): Promise<unknown> {
    return postJson(
        `/api/v1/deployments/${encodeURIComponent(deploymentId)}/gray`,
        {
            gray_version_id: grayVersionId,
            gray_traffic_ratio: grayTrafficRatio,
        },
    );
}

export async function fetchClusterNodes(): Promise<ClusterNode[]> {
    const data = await getJson<{ nodes: ClusterNode[] }>(
        "/api/v1/cluster/nodes",
    );
    return data.nodes ?? [];
}

export async function addClusterNode(input: {
    name: string;
    url: string;
    capabilities?: string;
}): Promise<unknown> {
    return sendJson("POST", "/api/v1/cluster/nodes", input);
}

export async function removeClusterNode(nodeId: string): Promise<void> {
    await sendJson(
        "DELETE",
        `/api/v1/cluster/nodes/${encodeURIComponent(nodeId)}`,
    );
}

export async function fetchVersions(modelId: string): Promise<VersionRecord[]> {
    const data = await getJson<{ items: VersionRecord[] }>(
        `/api/v1/models/${encodeURIComponent(modelId)}/versions?page=1&page_size=50`,
    );
    return data.items ?? [];
}

export async function versionAction(
    versionId: string,
    action: "rollback" | "promote" | "retire",
): Promise<unknown> {
    return postJson(
        `/api/v1/versions/${encodeURIComponent(versionId)}/${action}`,
    );
}

export async function fetchScans(): Promise<SecurityScan[]> {
    const data = await getJson<{ items: SecurityScan[] }>(
        "/api/v1/security/scans?page=1&page_size=50",
    );
    return data.items ?? [];
}

export async function triggerScan(
    modelId: string,
    versionId = "",
): Promise<SecurityScan> {
    return postJson("/api/v1/security/scan", {
        model_id: modelId,
        version_id: versionId,
    });
}

export async function fetchAuditLogs(): Promise<AuditLogEntry[]> {
    const data = await getJson<{ items: AuditLogEntry[] }>(
        "/api/v1/system/audit?page=1&page_size=50",
    );
    return data.items ?? [];
}
