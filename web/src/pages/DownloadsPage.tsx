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
    type MarketResult,
    type MarketSource,
    cancelDownload,
    createDownload,
    createModel,
    fetchDownloads,
    fetchModels,
    importFromHf,
    marketFiles,
    marketSearch,
    pickWeightFile,
} from "../api";
import { t, useLang } from "../i18n";
import { SpeedTracker, computeEtaMs, downloadSource } from "../telemetry";

// Download tasks (spec §7.3): progress, speed, ETA, source badge, cancellable.
// bug1: the entry point is SEARCH-first — query hf-mirror.com or modelscope.cn
// via /models/market/search, then a per-result Download action registers the
// model row and starts a single-file download of the repo's weight file
// (path resolved server-side via /models/market/files). Speed/ETA are derived
// client-side from successive poll samples. Cancel is a destructive op behind
// a danger confirm (spec S5).

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
    useLang();
    const [tasks, setTasks] = useState<DownloadTask[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [cancelId, setCancelId] = useState<string | null>(null);
    const [models, setModels] = useState<
        Array<{ id: string; name: string; hf_repo: string }>
    >([]);
    const [source, setSource] = useState<MarketSource>("hf");
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<MarketResult[] | null>(null);
    const [searching, setSearching] = useState(false);
    const [busyRepo, setBusyRepo] = useState("");
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
        fetchModels("", 1, 200)
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

    const onSearch = async () => {
        const kw = query.trim();
        if (!kw) return;
        setSearching(true);
        try {
            setResults(await marketSearch(kw, source));
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] market search fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
        setSearching(false);
    };

    const repoOf = (r: MarketResult): string =>
        source === "hf" ? r.id : [r.creator, r.name].filter(Boolean).join("/");

    const ensureModel = async (
        repo: string,
    ): Promise<{ id: string; name: string }> => {
        try {
            if (source === "hf") {
                return await importFromHf(repo);
            }
            return await createModel({
                name: repo.toLowerCase(),
                description: `Imported from ModelScope: ${repo}`,
            });
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            if (!/409|already exists/i.test(msg)) throw e;
        }
        const existing = await fetchModels(
            repo.split("/").pop() ?? repo,
            1,
            20,
        );
        const hit =
            existing.find((m) => m.hf_repo === repo) ??
            existing.find((m) => m.name === repo.toLowerCase()) ??
            existing.find((m) => m.name === repo);
        if (!hit) {
            throw new Error(
                t("模型记录未找到", "Model record not found after create"),
            );
        }
        return hit;
    };

    const onDownload = async (r: MarketResult) => {
        const repo = repoOf(r);
        if (!repo) return;
        setBusyRepo(repo);
        try {
            const model = await ensureModel(repo);
            const files = await marketFiles(source, repo);
            const file = pickWeightFile(files);
            if (!file) {
                throw new Error(
                    t(
                        "仓库内未找到权重文件",
                        "No weight file found in the repo",
                    ),
                );
            }
            const url =
                source === "hf"
                    ? `${HF_MIRROR}/${repo}/resolve/main/${file.path}`
                    : `https://modelscope.cn/models/${repo}/resolve/master/${file.path}`;
            await createDownload(model.id, url);
            setError(null);
            load();
        } catch (e) {
            console.error("[model-hub-web] download from search fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
        setBusyRepo("");
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
        return (
            <ErrorState
                message={error}
                retryLabel={t("重试", "Retry")}
                onRetry={load}
            />
        );
    }
    if (!tasks) {
        return <SkeletonLines rows={6} />;
    }

    const columns: DataTableColumn<DownloadTask>[] = [
        {
            key: "model_id",
            header: t("模型", "Model"),
            render: (r) => (
                <span className="font-mono text-xs text-text-primary">
                    {models.find((m) => m.id === r.model_id)?.name ??
                        r.model_id}
                </span>
            ),
        },
        {
            key: "source",
            header: t("来源", "Source"),
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
            header: t("进度", "Progress"),
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
            header: t("字节", "Bytes"),
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
            header: t("速度", "Speed"),
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
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge status={statusToKind(r.status)} label={r.status} />
            ),
        },
        {
            key: "actions",
            header: t("操作", "Actions"),
            render: (r) =>
                r.status === "pending" || r.status === "downloading" ? (
                    <button
                        type="button"
                        onClick={() => setCancelId(r.task_id)}
                        className="rounded-md border border-error/50 px-2 py-1 text-xs text-error hover:bg-error/10"
                    >
                        {t("取消", "Cancel")}
                    </button>
                ) : null,
        },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <h1 className="mb-3 text-base text-text-primary">
                {t("下载", "Downloads")}
            </h1>
            <div className="mb-4 rounded-card border border-border bg-bg-elevated p-3">
                <div className="mb-2 text-xs text-text-secondary">
                    {t(
                        "先搜索模型，再从 hf-mirror.com / modelscope.cn 下载",
                        "Search a model first, then download from hf-mirror.com / modelscope.cn",
                    )}
                </div>
                <div className="flex flex-wrap gap-2">
                    <select
                        name="market_source"
                        value={source}
                        onChange={(e) => {
                            setSource(e.target.value as MarketSource);
                            setResults(null);
                        }}
                        className="rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    >
                        <option value="hf">hf-mirror.com</option>
                        <option value="modelscope">modelscope.cn</option>
                    </select>
                    <input
                        name="market_query"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter") onSearch();
                        }}
                        placeholder={t(
                            "搜索模型，如 Qwen2.5-0.5B-Instruct",
                            "search models, e.g. Qwen2.5-0.5B-Instruct",
                        )}
                        className="min-w-64 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary outline-none focus:border-brand"
                    />
                    <button
                        type="button"
                        disabled={searching || !query.trim()}
                        onClick={onSearch}
                        className="rounded-md bg-brand px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {searching
                            ? t("搜索中…", "Searching…")
                            : t("搜索", "Search")}
                    </button>
                </div>
                {results !== null ? (
                    results.length === 0 ? (
                        <div className="mt-3 text-xs text-text-muted">
                            {t("无搜索结果", "No results")}
                        </div>
                    ) : (
                        <ul className="mt-3 divide-y divide-border">
                            {results.map((r) => {
                                const repo = repoOf(r);
                                return (
                                    <li
                                        key={`${r.source}-${repo}`}
                                        className="flex items-center gap-3 py-2"
                                    >
                                        <span className="min-w-0 flex-1 truncate font-mono text-xs text-text-primary">
                                            {repo}
                                        </span>
                                        {r.task ? (
                                            <span className="rounded-full border border-border px-2 py-0.5 text-[10px] text-text-muted">
                                                {r.task}
                                            </span>
                                        ) : null}
                                        <span className="font-mono text-[10px] tabular-nums text-text-muted">
                                            ↓ {r.downloads}
                                        </span>
                                        <button
                                            type="button"
                                            disabled={busyRepo === repo}
                                            onClick={() => onDownload(r)}
                                            className="rounded-md border border-brand/50 px-2 py-1 text-xs text-brand hover:bg-brand/10 disabled:cursor-not-allowed disabled:opacity-40"
                                        >
                                            {busyRepo === repo
                                                ? t("启动中…", "Starting…")
                                                : t("下载", "Download")}
                                        </button>
                                    </li>
                                );
                            })}
                        </ul>
                    )
                ) : null}
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
                        title={t("暂无下载任务", "No download tasks")}
                        description={t(
                            "在上方搜索模型并点击下载，任务进度会显示在这里。",
                            "Search a model above and hit Download; task progress shows up here.",
                        )}
                    />
                }
            />
            <ConfirmDialog
                open={cancelId !== null}
                danger
                title={
                    cancelId
                        ? t(
                              `取消下载 ${cancelId}？`,
                              `Cancel download ${cancelId}?`,
                          )
                        : ""
                }
                description={t(
                    "进行中的下载将被取消，未完成的 .part 文件会被丢弃。重试将从头开始。",
                    "The in-flight download is cancelled and the partial .part file is discarded. A retry restarts from zero.",
                )}
                confirmLabel={t("取消下载", "Cancel download")}
                cancelLabel={t("继续下载", "Keep downloading")}
                onConfirm={onCancel}
                onCancel={() => setCancelId(null)}
            />
        </div>
    );
}
