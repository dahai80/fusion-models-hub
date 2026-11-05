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
    type ApiKeyInfo,
    type ApprovalRequest,
    type Tenant,
    type Webhook,
    createApiKey,
    createTenant,
    createWebhook,
    deactivateApiKey,
    decideApproval,
    deleteWebhook,
    fetchApiKeys,
    fetchApprovals,
    fetchTenants,
    fetchWebhooks,
} from "../api";
import { t, useLang } from "../i18n";

// Governance console (issue #90): tenants, API keys, approval workflow,
// webhooks. Stacked read-mostly panels with the minimal write actions.

function boolToKind(active: boolean): StatusKind {
    return active ? "success" : "offline";
}

function Field({
    value,
    onChange,
    placeholder,
}: {
    value: string;
    onChange: (v: string) => void;
    placeholder: string;
}) {
    return (
        <input
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            className="min-w-32 flex-1 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
        />
    );
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

export function GovernancePage() {
    useLang();
    const [tenants, setTenants] = useState<Tenant[] | null>(null);
    const [keys, setKeys] = useState<ApiKeyInfo[] | null>(null);
    const [approvals, setApprovals] = useState<ApprovalRequest[] | null>(null);
    const [webhooks, setWebhooks] = useState<Webhook[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [newTenantName, setNewTenantName] = useState("");
    const [newKeyName, setNewKeyName] = useState("");
    const [newKeyRole, setNewKeyRole] = useState("developer");
    const [issuedKey, setIssuedKey] = useState("");
    const [approver, setApprover] = useState("");
    const [newHookName, setNewHookName] = useState("");
    const [newHookUrl, setNewHookUrl] = useState("");
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        try {
            const [ts, ks, aps, whs] = await Promise.all([
                fetchTenants(),
                fetchApiKeys(),
                fetchApprovals(),
                fetchWebhooks(),
            ]);
            setTenants(ts);
            setKeys(ks);
            setApprovals(aps);
            setWebhooks(whs);
            setError(null);
        } catch (e) {
            console.error("[model-hub-web] governance load fail:", e);
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
                setError(null);
            } catch (e) {
                console.error("[model-hub-web] governance action fail:", e);
                setError(e instanceof Error ? e.message : String(e));
            } finally {
                setBusy(false);
            }
        },
        [load],
    );

    if (error && !tenants) {
        return (
            <ErrorState
                message={error}
                retryLabel={t("重试", "Retry")}
                onRetry={load}
            />
        );
    }
    if (!tenants || !keys || !approvals || !webhooks) {
        return <SkeletonLines rows={8} />;
    }

    const tenantColumns: DataTableColumn<Tenant>[] = [
        { key: "name", header: t("名称", "Name") },
        { key: "display_name", header: t("显示名", "Display name") },
        {
            key: "is_active",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge
                    status={boolToKind(r.is_active)}
                    label={r.is_active ? "active" : "inactive"}
                />
            ),
        },
    ];

    const keyColumns: DataTableColumn<ApiKeyInfo>[] = [
        { key: "name", header: t("名称", "Name") },
        {
            key: "key_prefix",
            header: t("前缀", "Prefix"),
            render: (r) => (
                <span className="font-mono text-xs">{r.key_prefix}…</span>
            ),
        },
        { key: "role", header: t("角色", "Role") },
        { key: "tenant_id", header: "Tenant" },
        {
            key: "is_active",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge
                    status={boolToKind(r.is_active)}
                    label={r.is_active ? "active" : "inactive"}
                />
            ),
        },
        {
            key: "actions",
            header: "",
            render: (r) =>
                r.is_active ? (
                    <ActionButton
                        label={t("停用", "Deactivate")}
                        disabled={busy}
                        onClick={() => run(() => deactivateApiKey(r.id))}
                    />
                ) : null,
        },
    ];

    const approvalColumns: DataTableColumn<ApprovalRequest>[] = [
        { key: "model_id", header: t("模型", "Model") },
        { key: "version_id", header: t("版本", "Version") },
        { key: "level", header: t("级别", "Level") },
        {
            key: "status",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge
                    status={
                        r.status === "approved"
                            ? "success"
                            : r.status === "rejected"
                              ? "error"
                              : "warning"
                    }
                    label={r.status}
                />
            ),
        },
        {
            key: "actions",
            header: "",
            render: (r) =>
                r.status === "pending" ? (
                    <span className="flex gap-2">
                        <ActionButton
                            label={t("批准", "Approve")}
                            disabled={busy}
                            onClick={() =>
                                run(() =>
                                    decideApproval(r.id, "approve", approver),
                                )
                            }
                        />
                        <ActionButton
                            label={t("驳回", "Reject")}
                            disabled={busy}
                            onClick={() =>
                                run(() =>
                                    decideApproval(r.id, "reject", approver),
                                )
                            }
                        />
                    </span>
                ) : null,
        },
    ];

    const webhookColumns: DataTableColumn<Webhook>[] = [
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
        { key: "events", header: t("事件", "Events") },
        {
            key: "is_active",
            header: t("状态", "Status"),
            render: (r) => (
                <StatusBadge
                    status={boolToKind(r.is_active)}
                    label={r.is_active ? "active" : "inactive"}
                />
            ),
        },
        {
            key: "actions",
            header: "",
            render: (r) => (
                <ActionButton
                    label={t("删除", "Delete")}
                    disabled={busy}
                    onClick={() => run(() => deleteWebhook(r.id))}
                />
            ),
        },
    ];

    return (
        <div className="mx-auto max-w-5xl space-y-6 px-4">
            <h1 className="text-base text-text-primary">
                {t("治理", "Governance")}
            </h1>
            {error ? (
                <div className="rounded-md border border-error/40 bg-error/10 px-3 py-2 text-xs text-error">
                    {error}
                </div>
            ) : null}
            {issuedKey ? (
                <div className="rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-text-primary">
                    {t("新密钥 (仅显示一次):", "New key (shown once):")}{" "}
                    <span className="font-mono">{issuedKey}</span>
                </div>
            ) : null}

            <section>
                <h2 className="mb-2 text-sm text-text-secondary">
                    {t("租户", "Tenants")}
                </h2>
                <div className="mb-2 flex flex-wrap gap-2">
                    <Field
                        value={newTenantName}
                        onChange={setNewTenantName}
                        placeholder={t("租户名 (name)", "tenant name")}
                    />
                    <ActionButton
                        label={t("创建租户", "Create tenant")}
                        disabled={busy || !newTenantName.trim()}
                        onClick={() =>
                            run(async () => {
                                await createTenant(newTenantName.trim());
                                setNewTenantName("");
                            })
                        }
                    />
                </div>
                <DataTable
                    columns={tenantColumns}
                    rows={tenants}
                    rowKey={(r) => r.id}
                    pageSize={5}
                    emptyState={
                        <EmptyState
                            title={t("暂无租户", "No tenants")}
                            description={t(
                                "上方输入租户名创建。",
                                "Create one above by name.",
                            )}
                        />
                    }
                />
            </section>

            <section>
                <h2 className="mb-2 text-sm text-text-secondary">
                    {t("API 密钥", "API keys")}
                </h2>
                <div className="mb-2 flex flex-wrap gap-2">
                    <Field
                        value={newKeyName}
                        onChange={setNewKeyName}
                        placeholder={t("密钥名", "key name")}
                    />
                    <select
                        value={newKeyRole}
                        onChange={(e) => setNewKeyRole(e.target.value)}
                        className="rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-primary"
                    >
                        <option value="admin">admin</option>
                        <option value="developer">developer</option>
                        <option value="viewer">viewer</option>
                    </select>
                    <ActionButton
                        label={t("创建密钥", "Create key")}
                        disabled={busy || !newKeyName.trim()}
                        onClick={() =>
                            run(async () => {
                                const resp = await createApiKey({
                                    name: newKeyName.trim(),
                                    role: newKeyRole,
                                });
                                setIssuedKey(resp.key);
                                setNewKeyName("");
                            })
                        }
                    />
                </div>
                <DataTable
                    columns={keyColumns}
                    rows={keys}
                    rowKey={(r) => r.id}
                    pageSize={5}
                    emptyState={
                        <EmptyState
                            title={t("暂无密钥", "No API keys")}
                            description={t(
                                "上方创建第一个密钥。",
                                "Create the first key above.",
                            )}
                        />
                    }
                />
            </section>

            <section>
                <h2 className="mb-2 text-sm text-text-secondary">
                    {t("发布审批", "Publish approvals")}
                </h2>
                <div className="mb-2 flex flex-wrap gap-2">
                    <Field
                        value={approver}
                        onChange={setApprover}
                        placeholder={t("审批人签名", "approver signature")}
                    />
                </div>
                <DataTable
                    columns={approvalColumns}
                    rows={approvals}
                    rowKey={(r) => r.id}
                    pageSize={5}
                    emptyState={
                        <EmptyState
                            title={t("暂无审批单", "No approval requests")}
                        />
                    }
                />
            </section>

            <section>
                <h2 className="mb-2 text-sm text-text-secondary">
                    {t("Webhooks", "Webhooks")}
                </h2>
                <div className="mb-2 flex flex-wrap gap-2">
                    <Field
                        value={newHookName}
                        onChange={setNewHookName}
                        placeholder={t("名称", "name")}
                    />
                    <Field
                        value={newHookUrl}
                        onChange={setNewHookUrl}
                        placeholder="https://…"
                    />
                    <ActionButton
                        label={t("创建 Webhook", "Create webhook")}
                        disabled={
                            busy || !newHookName.trim() || !newHookUrl.trim()
                        }
                        onClick={() =>
                            run(async () => {
                                await createWebhook({
                                    name: newHookName.trim(),
                                    url: newHookUrl.trim(),
                                });
                                setNewHookName("");
                                setNewHookUrl("");
                            })
                        }
                    />
                </div>
                <DataTable
                    columns={webhookColumns}
                    rows={webhooks}
                    rowKey={(r) => r.id}
                    pageSize={5}
                    emptyState={
                        <EmptyState title={t("暂无 Webhook", "No webhooks")} />
                    }
                />
            </section>
        </div>
    );
}
