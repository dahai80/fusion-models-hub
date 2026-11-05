import {
    DataTable,
    type DataTableColumn,
    EmptyState,
    SkeletonLines,
    StatusBadge,
    type StatusKind,
} from "@fusion/web-ui";
import { useCallback, useEffect, useState } from "react";
import {
    type AuditLogEntry,
    type SecurityScan,
    type VersionRecord,
    fetchAuditLogs,
    fetchModels,
    fetchScans,
    fetchVersions,
    triggerScan,
    versionAction,
} from "../api";
import { t, useLang } from "../i18n";

// Versions & integrity console (issue #90): version lifecycle per model
// (promote/retire/rollback), security scans, audit log tail.

function vstatusToKind(status: string): StatusKind {
    switch (status) {
        case "published":
            return "success";
        case "draft":
        case "pending":
        case "scanning":
            return "warning";
        case "retired":
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

export function VersionsPage() {
    useLang();
    const [models, setModels] = useState<Array<{ id: string; name: string }>>(
        [],
    );
    const [modelId, setModelId] = useState("");
    const [versions, setVersions] = useState<VersionRecord[] | null>(null);
    const [scans, setScans] = useState<SecurityScan[] | null>(null);
    const [audit, setAudit] = useState<AuditLogEntry[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        try {
            const [ms, ss, logs] = await Promise.all([
                fetchModels(),
                fetchScans(),
                fetchAuditLogs(),
            ]);
            setModels(ms.map((m) => ({ id: m.id, name: m.name })));
            setScans(ss);
            setAudit(logs);
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] versions load fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    const loadVersions = useCallback(async (mid: string) => {
        if (!mid) {
            setVersions(null);
            return;
        }
        try {
            setVersions(await fetchVersions(mid));
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] versions load fail:", e);
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    useEffect(() => {
        load();
        const timer = setInterval(load, 30000);
        return () => clearInterval(timer);
    }, [load]);

    const run = useCallback(
        async (fn: () => Promise<unknown>) => {
            setBusy(true);
            try {
                await fn();
                await load();
                if (modelId) await loadVersions(modelId);
                setError(null);
            } catch (e) {
                console.error("[model-hub-web] versions action fail:", e);
                setError(e instanceof Error ? e.message : String(e));
            } finally {
                setBusy(false);
            }
        },
        [load, loadVersions, modelId],
    );

    const columns: DataTableColumn<VersionRecord>[] = [
        { key: "version", header: t("版本", "Version") },
        { key: "format", header: t("格式", "Format") },
        { key: "quantization", header: t("量化", "Quant") },
        {
            key: "status",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge
                    status={vstatusToKind(r.status)}
                    label={r.status}
                />
            ),
        },
        {
            key: "benchmark_score",
            header: t("评分", "Score"),
            render: (r) => (
                <span className="font-mono tabular-nums">
                    {r.benchmark_score ?? "-"}
                </span>
            ),
        },
        {
            key: "actions",
            header: "",
            render: (r) => (
                <span className="flex gap-2">
                    <ActionButton
                        label={t("发布", "Promote")}
                        disabled={busy || r.status === "published"}
                        onClick={() =>
                            run(() => versionAction(r.id, "promote"))
                        }
                    />
                    <ActionButton
                        label={t("回滚", "Rollback")}
                        disabled={busy}
                        onClick={() =>
                            run(() => versionAction(r.id, "rollback"))
                        }
                    />
                    <ActionButton
                        label={t("退役", "Retire")}
                        disabled={busy || r.status === "retired"}
                        onClick={() => run(() => versionAction(r.id, "retire"))}
                    />
                </span>
            ),
        },
    ];

    const scanColumns: DataTableColumn<SecurityScan>[] = [
        { key: "model_id", header: t("模型", "Model") },
        { key: "scan_type", header: t("类型", "Type") },
        {
            key: "status",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge
                    status={vstatusToKind(r.status)}
                    label={r.status}
                />
            ),
        },
        {
            key: "risk_level",
            header: t("风险", "Risk"),
            render: (r) => (
                <StatusBadge
                    status={
                        r.risk_level === "high"
                            ? "error"
                            : r.risk_level === "medium"
                              ? "warning"
                              : "success"
                    }
                    label={r.risk_level || "-"}
                />
            ),
        },
        { key: "created_at", header: t("时间", "Created"), sortable: true },
    ];

    const auditColumns: DataTableColumn<AuditLogEntry>[] = [
        { key: "action", header: t("操作", "Action") },
        { key: "resource_type", header: t("资源", "Resource") },
        {
            key: "resource_id",
            header: "ID",
            render: (r) => (
                <span className="font-mono text-xs">{r.resource_id}</span>
            ),
        },
        { key: "created_at", header: t("时间", "Time"), sortable: true },
    ];

    return (
        <div className="mx-auto max-w-5xl space-y-6 px-4">
            <h1 className="text-base text-text-primary">
                {t("版本与安全", "Versions & Security")}
            </h1>
            {error ? (
                <div className="rounded-md border border-error/40 bg-error/10 px-3 py-2 text-xs text-error">
                    {error}
                </div>
            ) : null}

            <section>
                <div className="mb-2 flex flex-wrap gap-2">
                    <select
                        value={modelId}
                        onChange={(e) => {
                            setModelId(e.target.value);
                            void loadVersions(e.target.value);
                        }}
                        className="min-w-64 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    >
                        <option value="">
                            {t("选择模型…", "select model…")}
                        </option>
                        {models.map((m) => (
                            <option key={m.id} value={m.id}>
                                {m.name}
                            </option>
                        ))}
                    </select>
                    <ActionButton
                        label={t("触发安全扫描", "Trigger scan")}
                        disabled={busy || !modelId}
                        onClick={() => run(() => triggerScan(modelId))}
                    />
                </div>
                {versions ? (
                    <DataTable
                        columns={columns}
                        rows={versions}
                        rowKey={(r) => r.id}
                        pageSize={10}
                        emptyState={
                            <EmptyState
                                title={t(
                                    "该模型暂无版本",
                                    "No versions for this model",
                                )}
                            />
                        }
                    />
                ) : (
                    <EmptyState
                        title={t(
                            "选择一个模型查看版本",
                            "Select a model to browse versions",
                        )}
                    />
                )}
            </section>

            <section>
                <h2 className="mb-2 text-sm text-text-secondary">
                    {t("安全扫描", "Security scans")}
                </h2>
                {scans ? (
                    <DataTable
                        columns={scanColumns}
                        rows={scans}
                        rowKey={(r) => r.id}
                        pageSize={5}
                        emptyState={
                            <EmptyState title={t("暂无扫描记录", "No scans")} />
                        }
                    />
                ) : (
                    <SkeletonLines rows={3} />
                )}
            </section>

            <section>
                <h2 className="mb-2 text-sm text-text-secondary">
                    {t("审计日志", "Audit log")}
                </h2>
                {audit ? (
                    <DataTable
                        columns={auditColumns}
                        rows={audit}
                        rowKey={(r) => r.id}
                        pageSize={10}
                        emptyState={
                            <EmptyState
                                title={t("暂无审计记录", "No audit entries")}
                            />
                        }
                    />
                ) : (
                    <SkeletonLines rows={3} />
                )}
            </section>
        </div>
    );
}
