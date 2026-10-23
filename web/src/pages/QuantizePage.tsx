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
    type QuantizeTask,
    fetchModel,
    fetchModels,
    fetchQuantizeTasks,
    submitQuantize,
} from "../api";

// Quantize task status (issue #56): submit form (source version / format /
// bits) + task table with live status; row click shows task detail
// (calibration dataset, output version, error message).

function statusToKind(status: string): StatusKind {
    switch (status) {
        case "completed":
            return "success";
        case "running":
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

interface VersionOption {
    version_id: string;
    label: string;
}

export function QuantizePage() {
    const [tasks, setTasks] = useState<QuantizeTask[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [versions, setVersions] = useState<VersionOption[]>([]);
    const [sourceVersionId, setSourceVersionId] = useState("");
    const [targetFormat, setTargetFormat] = useState("mlx");
    const [quantBits, setQuantBits] = useState(4);
    const [submitting, setSubmitting] = useState(false);
    const [selected, setSelected] = useState<QuantizeTask | null>(null);

    const load = useCallback(async () => {
        try {
            const list = await fetchQuantizeTasks();
            setTasks(list);
            setSelected((prev) =>
                prev ? (list.find((t) => t.id === prev.id) ?? prev) : null,
            );
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] quantize load fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    useEffect(() => {
        load();
        const t = setInterval(load, 5000);
        return () => clearInterval(t);
    }, [load]);

    useEffect(() => {
        // Load version options for the submit form: models with versions.
        fetchModels("", 1, 100)
            .then(async (items) => {
                const opts: VersionOption[] = [];
                for (const m of items) {
                    try {
                        const d = await fetchModel(m.id);
                        for (const v of d.versions) {
                            opts.push({
                                version_id: v.id,
                                label: `${m.name}:${v.version} (${v.format}/${v.quantization})`,
                            });
                        }
                    } catch (e) {
                        console.error(
                            "[model-hub-web] version list load fail:",
                            m.id,
                            e,
                        );
                    }
                }
                setVersions(opts);
            })
            .catch((e) =>
                console.error("[model-hub-web] model list for form fail:", e),
            );
    }, []);

    const onSubmit = async () => {
        if (!sourceVersionId) return;
        setSubmitting(true);
        try {
            await submitQuantize({
                source_version_id: sourceVersionId,
                target_format: targetFormat,
                quant_bits: quantBits,
            });
            setError(null);
            load();
        } catch (e) {
            console.error("[model-hub-web] quantize submit fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
        setSubmitting(false);
    };

    if (error && !tasks) {
        return <ErrorState message={error} retryLabel="Retry" onRetry={load} />;
    }
    if (!tasks) {
        return <SkeletonLines rows={6} />;
    }

    const columns: DataTableColumn<QuantizeTask>[] = [
        {
            key: "id",
            header: "Task",
            render: (r) => (
                <button
                    type="button"
                    onClick={() => setSelected(r)}
                    className={`font-mono text-xs ${
                        selected?.id === r.id
                            ? "text-brand"
                            : "text-text-primary underline-offset-2 hover:underline"
                    }`}
                >
                    {r.id}
                </button>
            ),
        },
        {
            key: "source_version_id",
            header: "Source Version",
            render: (r) => (
                <span className="font-mono text-xs text-text-secondary">
                    {r.source_version_id}
                </span>
            ),
        },
        { key: "target_format", header: "Format" },
        {
            key: "quant_bits",
            header: "Bits",
            render: (r) => (
                <span className="font-mono tabular-nums">{r.quant_bits}</span>
            ),
        },
        {
            key: "status",
            header: "Status",
            render: (r) => (
                <StatusBadge status={statusToKind(r.status)} label={r.status} />
            ),
        },
        { key: "created_at", header: "Created", sortable: true },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <h1 className="mb-3 text-base text-text-primary">Quantize</h1>
            <div className="mb-4 rounded-card border border-border bg-bg-elevated p-3">
                <div className="mb-2 text-xs text-text-secondary">
                    New quantize task (runs on Fusion-MLX)
                </div>
                <div className="flex flex-wrap gap-2">
                    <select
                        name="source_version_id"
                        value={sourceVersionId}
                        onChange={(e) => setSourceVersionId(e.target.value)}
                        className="min-w-64 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    >
                        <option value="">select source version…</option>
                        {versions.map((v) => (
                            <option key={v.version_id} value={v.version_id}>
                                {v.label}
                            </option>
                        ))}
                    </select>
                    <select
                        name="target_format"
                        value={targetFormat}
                        onChange={(e) => setTargetFormat(e.target.value)}
                        className="rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    >
                        <option value="mlx">mlx</option>
                        <option value="gguf">gguf</option>
                    </select>
                    <select
                        name="quant_bits"
                        value={quantBits}
                        onChange={(e) => setQuantBits(Number(e.target.value))}
                        className="rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    >
                        <option value={2}>2-bit</option>
                        <option value={4}>4-bit</option>
                        <option value={6}>6-bit</option>
                        <option value={8}>8-bit</option>
                    </select>
                    <button
                        type="button"
                        disabled={submitting || !sourceVersionId}
                        onClick={onSubmit}
                        className="rounded-md bg-brand px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {submitting ? "Submitting…" : "Submit"}
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
                rowKey={(r) => r.id}
                pageSize={20}
                emptyState={
                    <EmptyState
                        title="No quantize tasks"
                        description="Submit one above — quantization runs on the Fusion-MLX engine."
                    />
                }
            />
            {selected ? (
                <div className="mt-4">
                    <KVPanel
                        title={selected.id}
                        pairs={[
                            ["status", selected.status],
                            ["source_version_id", selected.source_version_id],
                            ["target_format", selected.target_format],
                            ["quant_bits", String(selected.quant_bits)],
                            [
                                "calibration_dataset",
                                selected.calibration_dataset || "-",
                            ],
                            [
                                "output_version_id",
                                selected.output_version_id || "-",
                            ],
                            ["error_message", selected.error_message || "-"],
                            ["created_at", selected.created_at ?? "-"],
                        ]}
                    />
                </div>
            ) : null}
        </div>
    );
}
