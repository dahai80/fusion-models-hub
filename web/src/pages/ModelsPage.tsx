import {
    DataTable,
    type DataTableColumn,
    EmptyState,
    ErrorState,
    KVPanel,
    SkeletonLines,
    StatusBadge,
    type StatusKind,
} from "@fusion/web-ui";
import { useCallback, useEffect, useState } from "react";
import {
    type ModelDetail,
    type ModelSummary,
    fetchModel,
    fetchModels,
    importFromHf,
} from "../api";

// Model repository browsing (issue #56): keyword search + table; row click
// loads the detail (versions, tags) into a side panel; HF import form
// registers a repo from the HF_MIRROR catalog without downloading.

function statusToKind(status: string): StatusKind {
    switch (status) {
        case "published":
            return "success";
        case "draft":
            return "offline";
        case "testing":
            return "running";
        case "deprecated":
        case "retired":
            return "warning";
        default:
            return "offline";
    }
}

export function ModelsPage() {
    const [models, setModels] = useState<ModelSummary[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [keyword, setKeyword] = useState("");
    const [query, setQuery] = useState("");
    const [detail, setDetail] = useState<ModelDetail | null>(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [hfRepo, setHfRepo] = useState("");
    const [importing, setImporting] = useState(false);
    const [notice, setNotice] = useState<string | null>(null);

    const load = useCallback(async () => {
        try {
            setModels(await fetchModels(query));
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] models load fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [query]);

    useEffect(() => {
        load();
        const t = setInterval(load, 30000);
        return () => clearInterval(t);
    }, [load]);

    const openDetail = async (modelId: string) => {
        setDetailLoading(true);
        try {
            setDetail(await fetchModel(modelId));
        } catch (e) {
            console.error("[model-hub-web] model detail load fail:", e);
            setDetail(null);
        }
        setDetailLoading(false);
    };

    const onImport = async () => {
        const repo = hfRepo.trim();
        if (!repo) return;
        setImporting(true);
        setNotice(null);
        try {
            const m = await importFromHf(repo);
            setNotice(
                `Imported ${m.name} (${m.id}) — start a download on the Downloads tab.`,
            );
            setHfRepo("");
            load();
        } catch (e) {
            console.error("[model-hub-web] hf import fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
        setImporting(false);
    };

    if (error && !models) {
        return <ErrorState message={error} retryLabel="Retry" onRetry={load} />;
    }
    if (!models) {
        return <SkeletonLines rows={6} />;
    }

    const columns: DataTableColumn<ModelSummary>[] = [
        {
            key: "name",
            header: "Model",
            sortable: true,
            render: (r) => (
                <button
                    type="button"
                    onClick={() => openDetail(r.id)}
                    className={`font-mono text-xs ${
                        detail?.id === r.id
                            ? "text-brand"
                            : "text-text-primary underline-offset-2 hover:underline"
                    }`}
                >
                    {r.name}
                </button>
            ),
        },
        { key: "model_type", header: "Type" },
        {
            key: "model_status",
            header: "Status",
            render: (r) => (
                <StatusBadge
                    status={statusToKind(r.model_status)}
                    label={r.model_status}
                />
            ),
        },
        { key: "versions_count", header: "Versions", sortable: true },
        { key: "params_size", header: "Params" },
        { key: "hf_repo", header: "HF Repo" },
        {
            key: "download_count",
            header: "Downloads",
            sortable: true,
            render: (r) => (
                <span className="font-mono tabular-nums">
                    {r.download_count}
                </span>
            ),
        },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <h1 className="mb-3 text-base text-text-primary">Models</h1>
            <div className="mb-4 flex flex-wrap items-center gap-2">
                <input
                    name="keyword"
                    value={keyword}
                    onChange={(e) => setKeyword(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") setQuery(keyword.trim());
                    }}
                    placeholder="search by name or description"
                    className="w-64 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary outline-none focus:border-brand"
                />
                <button
                    type="button"
                    onClick={() => setQuery(keyword.trim())}
                    className="rounded-md bg-brand px-4 py-2 text-sm text-white"
                >
                    Search
                </button>
            </div>
            <div className="mb-4 rounded-card border border-border bg-bg-elevated p-3">
                <div className="mb-2 text-xs text-text-secondary">
                    Import from Hugging Face (via hf-mirror.com, no download)
                </div>
                <div className="flex flex-wrap gap-2">
                    <input
                        name="hf_repo"
                        value={hfRepo}
                        onChange={(e) => setHfRepo(e.target.value)}
                        placeholder="org/repo (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
                        className="min-w-64 flex-1 rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-text-primary outline-none focus:border-brand"
                    />
                    <button
                        type="button"
                        disabled={importing || !hfRepo.trim()}
                        onClick={onImport}
                        className="rounded-md bg-brand px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {importing ? "Importing…" : "Import"}
                    </button>
                </div>
                {notice ? (
                    <div className="mt-2 rounded-md border border-success/40 bg-success/10 px-3 py-2 text-xs text-success">
                        {notice}
                    </div>
                ) : null}
            </div>
            {error ? (
                <div className="mb-2 rounded-md border border-error/40 bg-error/10 px-3 py-2 text-xs text-error">
                    {error}
                </div>
            ) : null}
            <DataTable
                columns={columns}
                rows={models}
                rowKey={(r) => r.id}
                pageSize={20}
                emptyState={
                    <EmptyState
                        title="No models yet"
                        description="Import a Hugging Face repo above to register your first model."
                    />
                }
            />
            {detailLoading ? (
                <div className="mt-4">
                    <SkeletonLines rows={4} />
                </div>
            ) : null}
            {detail ? (
                <div className="mt-4 rounded-card border border-border bg-bg-elevated p-3">
                    <div className="mb-2 flex items-center justify-between">
                        <span className="font-mono text-xs text-text-primary">
                            {detail.name}
                        </span>
                        <StatusBadge
                            status={statusToKind(detail.model_status)}
                            label={detail.model_status}
                        />
                    </div>
                    <KVPanel
                        pairs={[
                            ["id", detail.id],
                            ["type", detail.model_type],
                            ["architecture", detail.architecture || "-"],
                            ["params", detail.params_size || "-"],
                            ["license", detail.license || "-"],
                            ["hf_repo", detail.hf_repo || "-"],
                        ]}
                    />
                    <div className="mt-3 text-xs text-text-secondary">
                        Versions ({detail.versions.length})
                    </div>
                    <div className="mt-1 overflow-x-auto">
                        <table className="w-full text-left text-xs">
                            <thead>
                                <tr className="text-text-muted">
                                    <th className="py-1 pr-4 font-normal">
                                        Version
                                    </th>
                                    <th className="py-1 pr-4 font-normal">
                                        Format
                                    </th>
                                    <th className="py-1 pr-4 font-normal">
                                        Quant
                                    </th>
                                    <th className="py-1 pr-4 font-normal">
                                        Status
                                    </th>
                                    <th className="py-1 font-normal">Size</th>
                                </tr>
                            </thead>
                            <tbody>
                                {detail.versions.map((v) => (
                                    <tr
                                        key={v.id}
                                        className="border-t border-border"
                                    >
                                        <td className="py-1 pr-4 font-mono text-text-primary">
                                            {v.version}
                                        </td>
                                        <td className="py-1 pr-4">
                                            {v.format}
                                        </td>
                                        <td className="py-1 pr-4">
                                            {v.quantization}
                                        </td>
                                        <td className="py-1 pr-4">
                                            {v.status}
                                        </td>
                                        <td className="py-1 font-mono tabular-nums">
                                            {v.file_size}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            ) : null}
        </div>
    );
}
