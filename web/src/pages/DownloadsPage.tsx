import {
    ConfirmDialog,
    DataTable,
    type DataTableColumn,
    EmptyState,
    ErrorState,
    SkeletonLines,
    StatusBadge,
    type StatusKind,
    formatBytes,
    formatDuration,
} from "@fusion/web-ui";
import { useCallback, useEffect, useRef, useState } from "react";
import {
    type DownloadTask,
    HF_MIRROR,
    cancelDownload,
    createDownload,
    fetchDownloads,
    fetchModels,
} from "../api";
import { SpeedTracker, computeEtaMs, downloadSource } from "../telemetry";

// Download tasks (spec §7.3): progress, speed, ETA, HF_MIRROR source badge,
// cancellable. Speed/ETA are derived client-side from successive poll
// samples (the API reports cumulative bytes only). Cancel is a destructive
// op behind a danger confirm (spec S5).

function statusToKind(status: string): StatusKind {
    switch (status) {
        case "completed":
            return "success";
        case "downloading":
        case "pending":
            return "running";
        case "failed":
            return "error";
        case "cancelled":
            return "warning";
        default:
            return "offline";
    }
}

function SourceBadge({ sourceUrl }: { sourceUrl: string }) {
    const src = downloadSource(sourceUrl);
    const isMirror = src === "HF_MIRROR";
    return (
        <span
            className={`ml-2 rounded-full border px-2 py-0.5 text-[10px] ${
                isMirror
                    ? "border-brand/50 text-brand"
                    : "border-border text-text-muted"
            }`}
        >
            {src}
        </span>
    );
}

export function DownloadsPage() {
    const [tasks, setTasks] = useState<DownloadTask[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [cancelId, setCancelId] = useState<string | null>(null);
    const [models, setModels] = useState<
        Array<{ id: string; name: string; hf_repo: string }>
    >([]);
    const [modelId, setModelId] = useState("");
    const [hfRepo, setHfRepo] = useState("");
    const [submitting, setSubmitting] = useState(false);
    const trackerRef = useRef(new SpeedTracker());
    const speedsRef = useRef(new Map<string, number | null>());

    const load = useCallback(async () => {
        try {
            const list = await fetchDownloads();
            const tracker = trackerRef.current;
            const now = Date.now();
            for (const t of list) {
                if (t.status === "downloading") {
                    tracker.record(t.task_id, now, t.downloaded_bytes);
                } else {
                    tracker.reset(t.task_id);
                }
            }
            speedsRef.current = new Map(
                list.map((t) => [
                    t.task_id,
                    t.status === "downloading"
                        ? tracker.speed(t.task_id, now)
                        : null,
                ]),
            );
            setTasks(list);
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] downloads load fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    useEffect(() => {
        load();
        const t = setInterval(load, 5000);
        return () => clearInterval(t);
    }, [load]);

    useEffect(() => {
        fetchModels("", 1, 100)
            .then((items) =>
                setModels(
                    items.map((m) => ({
                        id: m.id,
                        name: m.name,
                        hf_repo: m.hf_repo,
                    })),
                ),
            )
            .catch((e) =>
                console.error("[model-hub-web] model list for form fail:", e),
            );
    }, []);

    const onSubmit = async () => {
        if (!modelId) return;
        const repo = hfRepo.trim();
        const selected = models.find((m) => m.id === modelId);
        const sourceUrl = repo
            ? `${HF_MIRROR}/${repo}`
            : selected?.hf_repo
              ? `${HF_MIRROR}/${selected.hf_repo}`
              : "";
        if (!sourceUrl) return;
        setSubmitting(true);
        try {
            await createDownload(modelId, sourceUrl);
            setHfRepo("");
            setError(null);
            load();
        } catch (e) {
            console.error("[model-hub-web] download create fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
        setSubmitting(false);
    };

    const onCancel = async () => {
        if (!cancelId) return;
        try {
            await cancelDownload(cancelId);
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] download cancel fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
        setCancelId(null);
        load();
    };

    if (error && !tasks) {
        return <ErrorState message={error} retryLabel="Retry" onRetry={load} />;
    }
    if (!tasks) {
        return <SkeletonLines rows={6} />;
    }

    const columns: DataTableColumn<DownloadTask>[] = [
        {
            key: "model_id",
            header: "Model",
            render: (r) => (
                <span className="font-mono text-xs text-text-primary">
                    {models.find((m) => m.id === r.model_id)?.name ??
                        r.model_id}
                </span>
            ),
        },
        {
            key: "source",
            header: "Source",
            render: (r) => (
                <span className="font-mono text-xs text-text-secondary">
                    {r.source_url.length > 40
                        ? `${r.source_url.slice(0, 40)}…`
                        : r.source_url}
                    <SourceBadge sourceUrl={r.source_url} />
                </span>
            ),
        },
        {
            key: "progress",
            header: "Progress",
            render: (r) => (
                <div className="min-w-32">
                    <div className="h-1.5 overflow-hidden rounded-full bg-bg-muted">
                        <div
                            className="h-full rounded-full bg-running transition-all"
                            style={{
                                width: `${Math.min(100, r.progress_percent)}%`,
                            }}
                        />
                    </div>
                    <div className="mt-1 font-mono text-[10px] tabular-nums text-text-muted">
                        {r.progress_percent.toFixed(1)}%
                    </div>
                </div>
            ),
        },
        {
            key: "bytes",
            header: "Bytes",
            render: (r) => (
                <span className="font-mono text-xs tabular-nums text-text-secondary">
                    {formatBytes(r.downloaded_bytes)}
                    {r.total_bytes > 0
                        ? ` / ${formatBytes(r.total_bytes)}`
                        : ""}
                </span>
            ),
        },
        {
            key: "speed",
            header: "Speed",
            render: (r) => {
                const speed = speedsRef.current.get(r.task_id) ?? null;
                const eta = computeEtaMs(
                    speed,
                    r.total_bytes - r.downloaded_bytes,
                );
                return (
                    <span className="font-mono text-xs tabular-nums text-text-secondary">
                        {speed !== null ? `${formatBytes(speed)}/s` : "-"}
                        {eta !== null ? (
                            <span className="block text-[10px] text-text-muted">
                                ETA {formatDuration(eta)}
                            </span>
                        ) : null}
                    </span>
                );
            },
        },
        {
            key: "status",
            header: "Status",
            render: (r) => (
                <StatusBadge status={statusToKind(r.status)} label={r.status} />
            ),
        },
        {
            key: "actions",
            header: "Actions",
            render: (r) =>
                r.status === "pending" || r.status === "downloading" ? (
                    <button
                        type="button"
                        onClick={() => setCancelId(r.task_id)}
                        className="rounded-md border border-error/50 px-2 py-1 text-xs text-error hover:bg-error/10"
                    >
                        Cancel
                    </button>
                ) : null,
        },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <h1 className="mb-3 text-base text-text-primary">Downloads</h1>
            <div className="mb-4 rounded-card border border-border bg-bg-elevated p-3">
                <div className="mb-2 text-xs text-text-secondary">
                    New download from hf-mirror.com
                </div>
                <div className="flex flex-wrap gap-2">
                    <select
                        name="model_id"
                        value={modelId}
                        onChange={(e) => setModelId(e.target.value)}
                        className="rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    >
                        <option value="">select model…</option>
                        {models.map((m) => (
                            <option key={m.id} value={m.id}>
                                {m.name}
                            </option>
                        ))}
                    </select>
                    <input
                        name="hf_repo"
                        value={hfRepo}
                        onChange={(e) => setHfRepo(e.target.value)}
                        placeholder="hf repo (e.g. Qwen/Qwen2.5-0.5B-Instruct); empty = model's hf_repo"
                        className="min-w-64 flex-1 rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-text-primary outline-none focus:border-brand"
                    />
                    <button
                        type="button"
                        disabled={
                            submitting ||
                            !modelId ||
                            (!hfRepo.trim() &&
                                !models.find((m) => m.id === modelId)?.hf_repo)
                        }
                        onClick={onSubmit}
                        className="rounded-md bg-brand px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {submitting ? "Starting…" : "Start download"}
                    </button>
                </div>
            </div>
            {error ? (
                <div className="mb-2 rounded-md border border-error/40 bg-error/10 px-3 py-2 text-xs text-error">
                    {error}
                </div>
            ) : null}
            <DataTable
                columns={columns}
                rows={tasks}
                rowKey={(r) => r.task_id}
                pageSize={20}
                emptyState={
                    <EmptyState
                        title="No download tasks"
                        description="Pick a model above to start downloading from hf-mirror.com."
                    />
                }
            />
            <ConfirmDialog
                open={cancelId !== null}
                danger
                title={cancelId ? `Cancel download ${cancelId}?` : ""}
                description="The in-flight download is cancelled and the partial .part file is discarded. A retry restarts from zero."
                confirmLabel="Cancel download"
                cancelLabel="Keep downloading"
                onConfirm={onCancel}
                onCancel={() => setCancelId(null)}
            />
        </div>
    );
}
