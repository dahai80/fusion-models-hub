import {
    ErrorState,
    KVPanel,
    MetricTile,
    SkeletonLines,
    StatusBadge,
    type StatusKind,
} from "@fusion/web-ui";
import { useCallback, useEffect, useState } from "react";
import {
    type HealthInfo,
    type StorageStats,
    fetchHealth,
    fetchStorage,
} from "../api";

// Storage usage (issue #56): storage stats tiles + health panel. Read-only
// view, 30s poll (spec §6.3 list cadence).

function healthToKind(status: string): StatusKind {
    switch (status) {
        case "healthy":
        case "ok":
            return "success";
        case "degraded":
            return "warning";
        case "unhealthy":
            return "error";
        default:
            return "offline";
    }
}

export function StoragePage() {
    const [stats, setStats] = useState<StorageStats | null>(null);
    const [health, setHealth] = useState<HealthInfo | null>(null);
    const [error, setError] = useState<string | null>(null);

    const load = useCallback(async () => {
        try {
            const [s, h] = await Promise.all([fetchStorage(), fetchHealth()]);
            setStats(s);
            setHealth(h);
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] storage load fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    useEffect(() => {
        load();
        const t = setInterval(load, 30000);
        return () => clearInterval(t);
    }, [load]);

    if (error && !stats) {
        return <ErrorState message={error} retryLabel="Retry" onRetry={load} />;
    }
    if (!stats) {
        return <SkeletonLines rows={4} />;
    }

    const mlxStatus =
        typeof health?.mlx === "object" && health?.mlx !== null
            ? String(health.mlx.status ?? "unknown")
            : "unknown";
    const mlxValue = (
        <span className="inline-flex items-center gap-2">
            <StatusBadge status={healthToKind(mlxStatus)} label={mlxStatus} />
            {typeof health?.mlx === "object" &&
            health?.mlx !== null &&
            typeof health.mlx.url === "string" ? (
                <span className="font-mono text-xs text-text-muted">
                    {health.mlx.url}
                </span>
            ) : null}
        </span>
    );

    return (
        <div className="mx-auto max-w-4xl px-4">
            <h1 className="mb-3 text-base text-text-primary">Storage</h1>
            {error ? (
                <div className="mb-2 rounded-md border border-error/40 bg-error/10 px-3 py-2 text-xs text-error">
                    {error}
                </div>
            ) : null}
            <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <MetricTile
                    label="Total size"
                    value={stats.total_size_gb}
                    unit="GiB"
                />
                <MetricTile label="Models" value={stats.model_count} />
                <MetricTile label="Files" value={stats.file_count} />
            </div>
            <KVPanel
                title="Backend"
                pairs={[
                    ["storage path", stats.path ?? stats.endpoint ?? "-"],
                    ["bucket", stats.bucket ?? "-"],
                    [
                        "data dir",
                        typeof health?.data_dir === "string"
                            ? health.data_dir
                            : "-",
                    ],
                    ["fusion-mlx", mlxValue],
                ]}
            />
        </div>
    );
}
