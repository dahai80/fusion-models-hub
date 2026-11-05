import {
    DataTable,
    type DataTableColumn,
    EmptyState,
    ErrorState,
    SkeletonLines,
    StatusBadge,
    type StatusKind,
} from "@fusion/web-ui";
import { useCallback, useEffect, useState } from "react";
import {
    type Deployment,
    createDeployment,
    fetchDeployments,
    scaleDeployment,
    setGrayRelease,
    stopDeployment,
} from "../api";
import { t, useLang } from "../i18n";

// Serving console (issue #90): deployment instances, scale, gray release,
// stop. Create form needs a model id; replicas 1-100 per API contract.

function statusToKind(status: string): StatusKind {
    switch (status) {
        case "running":
        case "active":
            return "success";
        case "pending":
        case "creating":
        case "gray":
            return "warning";
        case "stopped":
        case "failed":
            return "error";
        default:
            return "offline";
    }
}

function ActionButton({
    label,
    disabled,
    onClick,
}: {
    label: string;
    disabled?: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            disabled={disabled}
            onClick={onClick}
            className="rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
            {label}
        </button>
    );
}

export function ServingPage() {
    useLang();
    const [deployments, setDeployments] = useState<Deployment[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [modelId, setModelId] = useState("");
    const [name, setName] = useState("");
    const [replicas, setReplicas] = useState(1);
    const [grayRatio] = useState(10);
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        try {
            setDeployments(await fetchDeployments());
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] serving load fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    useEffect(() => {
        load();
        const timer = setInterval(load, 15000);
        return () => clearInterval(timer);
    }, [load]);

    const run = useCallback(
        async (fn: () => Promise<unknown>) => {
            setBusy(true);
            try {
                await fn();
                await load();
                setError(null);
            } catch (e) {
                console.error("[model-hub-web] serving action fail:", e);
                setError(e instanceof Error ? e.message : String(e));
            } finally {
                setBusy(false);
            }
        },
        [load],
    );

    if (error && !deployments) {
        return (
            <ErrorState
                message={error}
                retryLabel={t("重试", "Retry")}
                onRetry={load}
            />
        );
    }
    if (!deployments) {
        return <SkeletonLines rows={6} />;
    }

    const columns: DataTableColumn<Deployment>[] = [
        { key: "name", header: t("名称", "Name") },
        { key: "model_id", header: t("模型", "Model") },
        { key: "version_id", header: t("版本", "Version") },
        {
            key: "replicas",
            header: t("副本", "Replicas"),
            render: (r) => (
                <span className="font-mono tabular-nums">{r.replicas}</span>
            ),
        },
        { key: "node_id", header: t("节点", "Node") },
        {
            key: "status",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge status={statusToKind(r.status)} label={r.status} />
            ),
        },
        {
            key: "gray",
            header: t("灰度", "Gray"),
            render: (r) =>
                r.gray_enabled ? (
                    <StatusBadge
                        status="warning"
                        label={`${r.gray_traffic_ratio}%`}
                    />
                ) : (
                    <span className="text-xs text-text-muted">-</span>
                ),
        },
        {
            key: "actions",
            header: "",
            render: (r) => (
                <span className="flex gap-2">
                    <ActionButton
                        label={t("缩容至1", "Scale to 1")}
                        disabled={busy || r.replicas === 1}
                        onClick={() => run(() => scaleDeployment(r.id, 1))}
                    />
                    <ActionButton
                        label={t("灰度10%", "Gray 10%")}
                        disabled={busy || r.gray_enabled || !r.version_id}
                        onClick={() =>
                            run(() =>
                                setGrayRelease(r.id, r.version_id, grayRatio),
                            )
                        }
                    />
                    <ActionButton
                        label={t("停止", "Stop")}
                        disabled={busy}
                        onClick={() => run(() => stopDeployment(r.id))}
                    />
                </span>
            ),
        },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <h1 className="mb-3 text-base text-text-primary">
                {t("服务部署", "Serving")}
            </h1>
            <div className="mb-4 rounded-card border border-border bg-bg-elevated p-3">
                <div className="mb-2 text-xs text-text-secondary">
                    {t(
                        "新建部署 (加载到 Fusion-MLX)",
                        "New deployment (loads onto Fusion-MLX)",
                    )}
                </div>
                <div className="flex flex-wrap gap-2">
                    <input
                        value={modelId}
                        onChange={(e) => setModelId(e.target.value)}
                        placeholder={t("模型 ID", "model id")}
                        className="min-w-40 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    />
                    <input
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder={t(
                            "部署名 (可选)",
                            "deployment name (optional)",
                        )}
                        className="min-w-40 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    />
                    <input
                        type="number"
                        min={1}
                        max={100}
                        value={replicas}
                        onChange={(e) =>
                            setReplicas(Number(e.target.value) || 1)
                        }
                        className="w-20 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    />
                    <button
                        type="button"
                        disabled={busy || !modelId.trim()}
                        onClick={() =>
                            run(async () => {
                                await createDeployment({
                                    model_id: modelId.trim(),
                                    name: name.trim(),
                                    replicas,
                                });
                                setModelId("");
                                setName("");
                            })
                        }
                        className="rounded-md bg-brand px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {t("部署", "Deploy")}
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
                rows={deployments}
                rowKey={(r) => r.id}
                pageSize={20}
                emptyState={
                    <EmptyState
                        title={t("暂无部署", "No deployments")}
                        description={t(
                            "上方输入模型 ID 部署一个实例。",
                            "Deploy an instance by model id above.",
                        )}
                    />
                }
            />
        </div>
    );
}
