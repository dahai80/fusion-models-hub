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
    type ClusterNode,
    addClusterNode,
    fetchClusterNodes,
    removeClusterNode,
} from "../api";
import { t, useLang } from "../i18n";

// Cluster console (issue #90): registered inference nodes + heartbeat
// freshness. Add/remove node, submit-free view (distributed task submit is
// API-only for now — needs a model picker tied to real model ids).

function statusToKind(status: string): StatusKind {
    switch (status) {
        case "online":
        case "alive":
            return "success";
        case "stale":
        case "unknown":
            return "warning";
        case "offline":
        case "dead":
            return "error";
        default:
            return "offline";
    }
}

export function ClusterPage() {
    useLang();
    const [nodes, setNodes] = useState<ClusterNode[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [newName, setNewName] = useState("");
    const [newUrl, setNewUrl] = useState("");
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        try {
            setNodes(await fetchClusterNodes());
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] cluster load fail:", e);
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
                console.error("[model-hub-web] cluster action fail:", e);
                setError(e instanceof Error ? e.message : String(e));
            } finally {
                setBusy(false);
            }
        },
        [load],
    );

    if (error && !nodes) {
        return (
            <ErrorState
                message={error}
                retryLabel={t("重试", "Retry")}
                onRetry={load}
            />
        );
    }
    if (!nodes) {
        return <SkeletonLines rows={6} />;
    }

    const columns: DataTableColumn<ClusterNode>[] = [
        { key: "name", header: t("名称", "Name") },
        {
            key: "url",
            header: "URL",
            render: (r) => (
                <span className="font-mono text-xs text-text-secondary">
                    {r.url}
                </span>
            ),
        },
        {
            key: "status",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge status={statusToKind(r.status)} label={r.status} />
            ),
        },
        { key: "capabilities", header: t("能力", "Capabilities") },
        {
            key: "last_seen",
            header: t("最后心跳", "Last heartbeat"),
            render: (r) => (
                <span className="text-xs text-text-secondary">
                    {r.last_seen ?? "-"}
                </span>
            ),
        },
        {
            key: "actions",
            header: "",
            render: (r) => (
                <button
                    type="button"
                    disabled={busy}
                    onClick={() => run(() => removeClusterNode(r.id))}
                    className="rounded-md border border-border px-3 py-1.5 text-xs text-text-secondary disabled:opacity-40"
                >
                    {t("移除", "Remove")}
                </button>
            ),
        },
    ];

    return (
        <div className="mx-auto max-w-5xl px-4">
            <h1 className="mb-3 text-base text-text-primary">
                {t("集群节点", "Cluster")}
            </h1>
            <div className="mb-4 rounded-card border border-border bg-bg-elevated p-3">
                <div className="mb-2 text-xs text-text-secondary">
                    {t(
                        "注册推理节点 (直连 fusion-mlx)",
                        "Register an inference node (direct fusion-mlx)",
                    )}
                </div>
                <div className="flex flex-wrap gap-2">
                    <input
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        placeholder={t("节点名", "node name")}
                        className="min-w-32 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    />
                    <input
                        value={newUrl}
                        onChange={(e) => setNewUrl(e.target.value)}
                        placeholder="http://host:port"
                        className="min-w-48 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    />
                    <button
                        type="button"
                        disabled={busy || !newName.trim() || !newUrl.trim()}
                        onClick={() =>
                            run(async () => {
                                await addClusterNode({
                                    name: newName.trim(),
                                    url: newUrl.trim(),
                                });
                                setNewName("");
                                setNewUrl("");
                            })
                        }
                        className="rounded-md bg-brand px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {t("注册", "Register")}
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
                rows={nodes}
                rowKey={(r) => r.id}
                pageSize={20}
                emptyState={
                    <EmptyState
                        title={t("暂无节点", "No nodes")}
                        description={t(
                            "注册一个 fusion-mlx 节点组成集群。",
                            "Register a fusion-mlx node to form a cluster.",
                        )}
                    />
                }
            />
            <div className="mt-4">
                <KVPanel
                    title={t("说明", "Notes")}
                    pairs={[
                        [
                            t("心跳", "Heartbeat"),
                            t(
                                "节点定期上报; 超时标记 stale, 由后端清理。",
                                "Nodes report periodically; stale ones are marked and reaped server-side.",
                            ),
                        ],
                        [
                            t("分布式任务", "Distributed tasks"),
                            t(
                                "提交走 POST /api/v1/cluster/distributed-tasks (UI 后续版本)。",
                                "Submit via POST /api/v1/cluster/distributed-tasks (UI in a later pass).",
                            ),
                        ],
                    ]}
                />
            </div>
        </div>
    );
}
