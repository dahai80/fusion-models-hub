import hashlib
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    ApiKey,
    ApprovalLevel,
    ApprovalRequest,
    AuditLog,
    BranchStatus,
    ClusterNode,
    Deployment,
    DistributedTask,
    DownloadTask,
    EvaluationResult,
    GitLfsLock,
    LoraMergeTask,
    Model,
    ModelBranch,
    ModelFavorite,
    ModelFormat,
    ModelRating,
    ModelTag,
    ModelType,
    ModelVersion,
    Quantization,
    QuantizeTask,
    Role,
    SecurityScan,
    Tenant,
    VersionStatus,
    Watermark,
    Webhook,
)

logger = logging.getLogger(__name__)

MAX_PAGE_SIZE = 200


async def create_model(
    session: AsyncSession,
    *,
    name: str,
    tenant_id: str = "",
    description: str = "",
    model_type: ModelType = ModelType.LLM,
    base_model_id: str = "",
    architecture: str = "",
    params_size: str = "",
    license: str = "",
    author: str = "",
    language: str = "",
    task_types: str = "",
    owner: str = "",
    hf_repo: str = "",
    model_modules: str = "",
    idle_timeout_minutes: int = 60,
) -> Model:
    m = Model(
        name=name, tenant_id=tenant_id, description=description, model_type=model_type,
        base_model_id=base_model_id, architecture=architecture, params_size=params_size,
        license=license, author=author, language=language,
        task_types=task_types, owner=owner, hf_repo=hf_repo,
        model_modules=model_modules, idle_timeout_minutes=idle_timeout_minutes,
    )
    session.add(m)
    await session.commit()
    await session.refresh(m)
    logger.info("Created model: id=%s name=%s", m.id, m.name)
    return m


async def get_model(session: AsyncSession, model_id: str) -> Model | None:
    result = await session.execute(select(Model).where(Model.id == model_id))
    return result.scalar_one_or_none()


async def get_model_by_name(session: AsyncSession, name: str) -> Model | None:
    result = await session.execute(select(Model).where(Model.name == name))
    return result.scalar_one_or_none()


async def list_models(
    session: AsyncSession,
    *,
    tenant_id: str = "",
    keyword: str = "",
    model_type: str = "",
    architecture: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Model], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    query = select(Model)
    count_query = select(func.count()).select_from(Model)

    if tenant_id:
        query = query.where(Model.tenant_id == tenant_id)
        count_query = count_query.where(Model.tenant_id == tenant_id)
    if keyword:
        safe_keyword = keyword[:64].replace("%", "\\%").replace("_", "\\_")
        cond = (
            Model.name.ilike(f"%{safe_keyword}%", escape="\\")
            | Model.description.ilike(f"%{safe_keyword}%", escape="\\")
            | Model.architecture.ilike(f"%{safe_keyword}%", escape="\\")
        )
        query = query.where(cond)
        count_query = count_query.where(cond)
    if model_type:
        query = query.where(Model.model_type == model_type)
        count_query = count_query.where(Model.model_type == model_type)
    if architecture:
        query = query.where(Model.architecture == architecture)
        count_query = count_query.where(Model.architecture == architecture)

    total = (await session.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(Model.updated_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    return list(result.scalars().all()), total


_MODEL_UPDATABLE = {
    "description", "model_type", "base_model_id", "architecture", "params_size",
    "license", "author", "language", "task_types", "owner", "hf_repo",
    "model_modules", "idle_timeout_minutes", "ttl_seconds", "pinned", "model_status",
}


async def update_model(session: AsyncSession, model_id: str, **fields) -> Model | None:
    m = await get_model(session, model_id)
    if not m:
        return None
    for k, v in fields.items():
        if k in _MODEL_UPDATABLE and v is not None:
            setattr(m, k, v)
    await session.commit()
    await session.refresh(m)
    logger.info("Updated model: id=%s fields=%s", model_id, list(fields.keys()))
    return m


async def list_pinned_models(session: AsyncSession) -> list[Model]:
    result = await session.execute(
        select(Model).where(Model.pinned.is_(True)).order_by(Model.updated_at.desc())
    )
    return list(result.scalars().all())


async def delete_model(session: AsyncSession, model_id: str) -> bool:
    m = await get_model(session, model_id)
    if not m:
        return False
    await session.delete(m)
    await session.commit()
    logger.info("Deleted model: id=%s", model_id)
    return True


async def increment_download(session: AsyncSession, model_id: str) -> None:
    await session.execute(
        update(Model).where(Model.id == model_id).values(download_count=Model.download_count + 1)
    )
    await session.commit()


# -- Version CRUD --

async def create_version(
    session: AsyncSession,
    *,
    model_id: str,
    version: str,
    format: ModelFormat = ModelFormat.MLX,
    quantization: Quantization = Quantization.Q4,
    file_path: str = "",
    file_hash: str = "",
    file_size: int = 0,
    release_notes: str = "",
) -> ModelVersion | None:
    m = await get_model(session, model_id)
    if not m:
        return None
    v = ModelVersion(
        model_id=model_id, version=version, format=format,
        quantization=quantization, file_path=file_path,
        file_hash=file_hash, file_size=file_size,
        release_notes=release_notes,
    )
    session.add(v)
    await session.commit()
    await session.refresh(v)
    logger.info("Created version: id=%s model=%s ver=%s", v.id, model_id, version)
    return v


async def get_version(session: AsyncSession, version_id: str) -> ModelVersion | None:
    result = await session.execute(select(ModelVersion).where(ModelVersion.id == version_id))
    return result.scalar_one_or_none()


async def list_versions(
    session: AsyncSession,
    model_id: str,
    *,
    tenant_id: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[ModelVersion], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    query = select(ModelVersion).where(ModelVersion.model_id == model_id)
    count_query = select(func.count()).select_from(ModelVersion).where(ModelVersion.model_id == model_id)

    if tenant_id:
        query = query.join(Model, ModelVersion.model_id == Model.id).where(Model.tenant_id == tenant_id)
        count_query = count_query.join(Model, ModelVersion.model_id == Model.id).where(Model.tenant_id == tenant_id)
    if status:
        query = query.where(ModelVersion.status == status)
        count_query = count_query.where(ModelVersion.status == status)

    total = (await session.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(ModelVersion.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    return list(result.scalars().all()), total


VALID_TRANSITIONS: dict[VersionStatus, set[VersionStatus]] = {
    VersionStatus.DRAFT: {VersionStatus.TESTING, VersionStatus.RETIRED},
    VersionStatus.TESTING: {VersionStatus.PUBLISHED, VersionStatus.DRAFT, VersionStatus.RETIRED},
    VersionStatus.PUBLISHED: {VersionStatus.DEPRECATED, VersionStatus.RETIRED},
    VersionStatus.DEPRECATED: {VersionStatus.PUBLISHED, VersionStatus.RETIRED},
    VersionStatus.RETIRED: set(),
}

EVALUATION_THRESHOLDS: dict[str, float] = {
    "l1": 50.0,
    "l2": 70.0,
    "l3": 85.0,
}


class InvalidTransition(Exception):
    pass


class EvaluationThresholdError(Exception):
    pass


def check_evaluation_threshold(version: ModelVersion, approval_level: str = "l1") -> None:
    threshold = EVALUATION_THRESHOLDS.get(approval_level, 0.0)
    if version.benchmark_score < threshold:
        raise EvaluationThresholdError(
            f"Benchmark score {version.benchmark_score} below threshold {threshold} for level {approval_level}"
        )


async def update_version_status(
    session: AsyncSession, version_id: str, target_status: VersionStatus,
    approval_level: str = "l1",
) -> ModelVersion | None:
    result = await session.execute(
        select(ModelVersion).where(ModelVersion.id == version_id).with_for_update()
    )
    v = result.scalar_one_or_none()
    if not v:
        return None
    allowed = VALID_TRANSITIONS.get(v.status, set())
    if target_status not in allowed:
        raise InvalidTransition(
            f"Cannot transition from {v.status.value} to {target_status.value}"
        )
    if target_status == VersionStatus.PUBLISHED:
        check_evaluation_threshold(v, approval_level)
    v.status = target_status
    await session.commit()
    await session.refresh(v)
    logger.info("Version status changed: id=%s -> %s", version_id, target_status.value)
    return v


_VERSION_UPDATABLE = {
    "file_path", "file_hash", "file_size", "release_notes",
    "benchmark_score", "inference_latency", "throughput", "memory_usage",
    "context_length", "successor_version_id", "encrypted",
    "license_type", "data_compliance",
}


async def update_version(session: AsyncSession, version_id: str, **fields) -> ModelVersion | None:
    v = await get_version(session, version_id)
    if not v:
        return None
    for k, val in fields.items():
        if k in _VERSION_UPDATABLE and val is not None:
            setattr(v, k, val)
    await session.commit()
    await session.refresh(v)
    return v


# -- Tag CRUD --

async def set_tags(session: AsyncSession, model_id: str, tags: list[dict[str, str]]) -> list[ModelTag]:
    await session.execute(delete(ModelTag).where(ModelTag.model_id == model_id))
    new_tags = []
    for t in tags:
        tag = ModelTag(model_id=model_id, key=t.get("key", ""), value=t.get("value", ""))
        session.add(tag)
        new_tags.append(tag)
    await session.commit()
    for tag in new_tags:
        await session.refresh(tag)
    logger.info("Set tags for model %s: count=%d", model_id, len(new_tags))
    return new_tags


# -- QuantizeTask CRUD --

async def create_quantize_task(
    session: AsyncSession,
    *,
    source_version_id: str,
    target_format: str = "mlx",
    quant_bits: int = 4,
    calibration_dataset: str = "",
) -> QuantizeTask:
    t = QuantizeTask(
        source_version_id=source_version_id,
        target_format=target_format,
        quant_bits=quant_bits,
        calibration_dataset=calibration_dataset,
    )
    session.add(t)
    await session.commit()
    await session.refresh(t)
    logger.info("Created quantize task: id=%s source=%s bits=%d", t.id, source_version_id, quant_bits)
    return t


async def get_quantize_task(session: AsyncSession, task_id: str) -> QuantizeTask | None:
    result = await session.execute(select(QuantizeTask).where(QuantizeTask.id == task_id))
    return result.scalar_one_or_none()


async def list_quantize_tasks(
    session: AsyncSession,
    *,
    status: str = "",
    page: int = 1,
    page_size: int = 20,
    tenant_id: str = "",
) -> tuple[list[QuantizeTask], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    query = select(QuantizeTask)
    count_query = select(func.count()).select_from(QuantizeTask)
    if status:
        query = query.where(QuantizeTask.status == status)
        count_query = count_query.where(QuantizeTask.status == status)
    if tenant_id:
        # F-04: scope quantize tasks to caller tenant via source version -> model.
        query = query.join(ModelVersion, QuantizeTask.source_version_id == ModelVersion.id).join(
            Model, ModelVersion.model_id == Model.id
        ).where(Model.tenant_id == tenant_id)
        count_query = count_query.join(ModelVersion, QuantizeTask.source_version_id == ModelVersion.id).join(
            Model, ModelVersion.model_id == Model.id
        ).where(Model.tenant_id == tenant_id)
    total = (await session.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(QuantizeTask.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    return list(result.scalars().all()), total


async def quantize_task_tenant(session: AsyncSession, task_id: str) -> str:
    # F-04: resolve a quantize task's owning tenant (via source version -> model).
    task = await get_quantize_task(session, task_id)
    if not task:
        return ""
    ver = await get_version(session, task.source_version_id)
    if not ver:
        return ""
    m = await get_model(session, ver.model_id)
    return m.tenant_id if m else ""


_TASK_UPDATABLE = {"status", "output_version_id", "error_message", "started_at", "completed_at", "calibration_dataset"}


async def update_quantize_task(
    session: AsyncSession,
    task_id: str,
    **fields,
) -> QuantizeTask | None:
    t = await get_quantize_task(session, task_id)
    if not t:
        return None
    for k, val in fields.items():
        if k in _TASK_UPDATABLE and val is not None:
            setattr(t, k, val)
    await session.commit()
    await session.refresh(t)
    logger.info("Updated quantize task: id=%s fields=%s", task_id, list(fields.keys()))
    return t


# -- ApiKey CRUD --


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _generate_api_key() -> tuple[str, str, str]:
    raw = uuid.uuid4().hex + uuid.uuid4().hex
    prefix = f"fmh-{raw[:4]}"
    full_key = f"{prefix}-{raw[4:]}"
    key_hash = _hash_key(full_key)
    return full_key, key_hash, prefix


async def create_api_key(
    session: AsyncSession,
    *,
    name: str,
    tenant_id: str = "",
    permissions: str = "read,write",
    role: str = "developer",
    qps_limit: int = 0,
    allowed_models: str = "",
    allowed_modules: str = "",
) -> tuple[ApiKey, str]:
    full_key, key_hash, key_prefix = _generate_api_key()
    from .models import UserRole
    ak = ApiKey(
        name=name, tenant_id=tenant_id, key_hash=key_hash,
        key_prefix=key_prefix, permissions=permissions, role=UserRole(role),
        qps_limit=qps_limit, allowed_models=allowed_models,
        allowed_modules=allowed_modules,
    )
    session.add(ak)
    await session.commit()
    await session.refresh(ak)
    logger.info("Created API key: id=%s name=%s", ak.id, name)
    return ak, full_key


_API_KEY_UPDATABLE = {"permissions", "role", "qps_limit", "allowed_models", "allowed_modules"}


async def get_api_key(session: AsyncSession, key_id: str) -> ApiKey | None:
    result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
    return result.scalar_one_or_none()


async def list_api_keys(session: AsyncSession, *, tenant_id: str = "") -> list[ApiKey]:
    stmt = select(ApiKey)
    if tenant_id:
        stmt = stmt.where(ApiKey.tenant_id == tenant_id)
    stmt = stmt.order_by(ApiKey.created_at.desc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def count_active_api_keys(session: AsyncSession) -> int:
    from sqlalchemy import func
    result = await session.execute(
        select(func.count(ApiKey.id)).where(ApiKey.is_active.is_(True))
    )
    return int(result.scalar() or 0)


async def touch_api_key_last_used(session: AsyncSession, key_id: str) -> None:
    # F-08: throttled background refresh of last_used_at, decoupled from verify.
    result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
    ak = result.scalar_one_or_none()
    if ak:
        ak.last_used_at = datetime.now(UTC)
        await session.commit()


async def verify_api_key(session: AsyncSession, raw_key: str) -> ApiKey | None:
    # F-08: pure read — no with_for_update, no last_used_at commit per request.
    # last_used_at refreshed via touch_api_key_last_used from the middleware on a
    # throttle; SQLite write lock no longer serializes every authenticated call.
    key_hash = _hash_key(raw_key)
    result = await session.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
    )
    ak = result.scalar_one_or_none()
    if ak:
        logger.info("API key verified: id=%s name=%s", ak.id, ak.name)
    return ak


async def deactivate_api_key(session: AsyncSession, key_id: str) -> ApiKey | None:
    ak = await get_api_key(session, key_id)
    if not ak:
        return None
    ak.is_active = False
    await session.commit()
    await session.refresh(ak)
    logger.info("Deactivated API key: id=%s", key_id)
    return ak


async def delete_api_key(session: AsyncSession, key_id: str) -> bool:
    ak = await get_api_key(session, key_id)
    if not ak:
        return False
    await session.delete(ak)
    await session.commit()
    logger.info("Deleted API key: id=%s", key_id)
    return True


# -- AuditLog CRUD --

async def create_audit_log(
    session: AsyncSession,
    *,
    action: str,
    resource_type: str,
    resource_id: str = "",
    api_key_id: str = "",
    tenant_id: str = "",
    detail: str = "",
) -> AuditLog:
    log = AuditLog(
        action=action, resource_type=resource_type,
        resource_id=resource_id, api_key_id=api_key_id,
        tenant_id=tenant_id, detail=detail,
    )
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log


async def list_audit_logs(
    session: AsyncSession,
    *,
    tenant_id: str = "",
    resource_type: str = "",
    action: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[AuditLog], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    query = select(AuditLog)
    count_query = select(func.count()).select_from(AuditLog)
    if tenant_id:
        query = query.where(AuditLog.tenant_id == tenant_id)
        count_query = count_query.where(AuditLog.tenant_id == tenant_id)
    if resource_type:
        query = query.where(AuditLog.resource_type == resource_type)
        count_query = count_query.where(AuditLog.resource_type == resource_type)
    if action:
        query = query.where(AuditLog.action == action)
        count_query = count_query.where(AuditLog.action == action)
    total = (await session.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    return list(result.scalars().all()), total


# -- Tenant CRUD --

async def create_tenant(
    session: AsyncSession,
    *,
    name: str,
    display_name: str = "",
) -> Tenant:
    t = Tenant(name=name, display_name=display_name)
    session.add(t)
    await session.commit()
    await session.refresh(t)
    logger.info("Created tenant: id=%s name=%s", t.id, name)
    return t


async def get_tenant(session: AsyncSession, tenant_id: str) -> Tenant | None:
    result = await session.execute(select(Tenant).where(Tenant.id == tenant_id))
    return result.scalar_one_or_none()


async def get_tenant_by_name(session: AsyncSession, name: str) -> Tenant | None:
    result = await session.execute(select(Tenant).where(Tenant.name == name))
    return result.scalar_one_or_none()


async def list_tenants(session: AsyncSession) -> list[Tenant]:
    result = await session.execute(select(Tenant).order_by(Tenant.created_at.desc()))
    return list(result.scalars().all())


async def delete_tenant(session: AsyncSession, tenant_id: str) -> bool:
    t = await get_tenant(session, tenant_id)
    if not t:
        return False
    await session.delete(t)
    await session.commit()
    logger.info("Deleted tenant: id=%s", tenant_id)
    return True


# -- Role CRUD --

async def create_role(
    session: AsyncSession,
    *,
    tenant_id: str,
    name: str,
    permissions: str = "read",
) -> Role:
    r = Role(tenant_id=tenant_id, name=name, permissions=permissions)
    session.add(r)
    await session.commit()
    await session.refresh(r)
    logger.info("Created role: id=%s tenant=%s name=%s", r.id, tenant_id, name)
    return r


async def get_role(session: AsyncSession, role_id: str) -> Role | None:
    result = await session.execute(select(Role).where(Role.id == role_id))
    return result.scalar_one_or_none()


async def list_roles(session: AsyncSession, tenant_id: str) -> list[Role]:
    result = await session.execute(
        select(Role).where(Role.tenant_id == tenant_id).order_by(Role.created_at.desc())
    )
    return list(result.scalars().all())


async def update_role(
    session: AsyncSession,
    role_id: str,
    *,
    name: str | None = None,
    permissions: str | None = None,
    is_active: bool | None = None,
) -> Role | None:
    r = await get_role(session, role_id)
    if not r:
        return None
    if name is not None:
        r.name = name
    if permissions is not None:
        r.permissions = permissions
    if is_active is not None:
        r.is_active = is_active
    r.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(r)
    logger.info("Updated role: id=%s", role_id)
    return r


async def delete_role(session: AsyncSession, role_id: str) -> bool:
    r = await get_role(session, role_id)
    if not r:
        return False
    await session.delete(r)
    await session.commit()
    logger.info("Deleted role: id=%s", role_id)
    return True


# -- Webhook CRUD --

async def create_webhook(
    session: AsyncSession,
    *,
    name: str,
    url: str,
    tenant_id: str = "",
    secret: str = "",
    events: str = "",
) -> Webhook:
    w = Webhook(name=name, url=url, tenant_id=tenant_id, secret=secret, events=events)
    session.add(w)
    await session.commit()
    await session.refresh(w)
    logger.info("Created webhook: id=%s name=%s url=%s", w.id, name, url)
    return w


async def get_webhook(session: AsyncSession, webhook_id: str) -> Webhook | None:
    result = await session.execute(select(Webhook).where(Webhook.id == webhook_id))
    return result.scalar_one_or_none()


async def list_webhooks(session: AsyncSession, tenant_id: str = "") -> list[Webhook]:
    query = select(Webhook).order_by(Webhook.created_at.desc())
    if tenant_id:
        query = query.where(Webhook.tenant_id == tenant_id)
    result = await session.execute(query)
    return list(result.scalars().all())


async def delete_webhook(session: AsyncSession, webhook_id: str) -> bool:
    w = await get_webhook(session, webhook_id)
    if not w:
        return False
    await session.delete(w)
    await session.commit()
    logger.info("Deleted webhook: id=%s", webhook_id)
    return True


# -- Deployment CRUD --

async def create_deployment(
    session: AsyncSession,
    *,
    model_id: str,
    name: str,
    tenant_id: str = "",
    version_id: str = "",
    replicas: int = 1,
) -> Deployment:
    d = Deployment(
        model_id=model_id, name=name, tenant_id=tenant_id,
        version_id=version_id, replicas=replicas,
    )
    session.add(d)
    await session.commit()
    await session.refresh(d)
    logger.info("Created deployment: id=%s name=%s model=%s", d.id, name, model_id)
    return d


async def get_deployment(session: AsyncSession, deployment_id: str) -> Deployment | None:
    result = await session.execute(select(Deployment).where(Deployment.id == deployment_id))
    return result.scalar_one_or_none()


async def list_deployments(
    session: AsyncSession,
    tenant_id: str = "",
    model_id: str = "",
    status: str = "",
) -> list[Deployment]:
    query = select(Deployment).order_by(Deployment.created_at.desc())
    if tenant_id:
        query = query.where(Deployment.tenant_id == tenant_id)
    if model_id:
        query = query.where(Deployment.model_id == model_id)
    if status:
        query = query.where(Deployment.status == status)
    result = await session.execute(query)
    return list(result.scalars().all())


_DEPLOYMENT_UPDATABLE = {"replicas", "status", "version_id", "gray_enabled", "gray_version_id", "gray_traffic_ratio"}


async def update_deployment(session: AsyncSession, deployment_id: str, **fields) -> Deployment | None:
    d = await get_deployment(session, deployment_id)
    if not d:
        return None
    for k, val in fields.items():
        if k in _DEPLOYMENT_UPDATABLE and val is not None:
            setattr(d, k, val)
    await session.commit()
    await session.refresh(d)
    logger.info("Updated deployment: id=%s fields=%s", deployment_id, list(fields.keys()))
    return d


async def delete_deployment(session: AsyncSession, deployment_id: str) -> bool:
    d = await get_deployment(session, deployment_id)
    if not d:
        return False
    await session.delete(d)
    await session.commit()
    logger.info("Deleted deployment: id=%s", deployment_id)
    return True


# -- ClusterNode CRUD --

async def create_cluster_node(
    session: AsyncSession,
    *,
    name: str,
    url: str,
    capabilities: str = "inference,quantize",
) -> ClusterNode:
    node = ClusterNode(name=name, url=url, capabilities=capabilities)
    session.add(node)
    await session.commit()
    await session.refresh(node)
    logger.info("Created cluster node: id=%s name=%s url=%s", node.id, name, url)
    return node


async def get_cluster_node(session: AsyncSession, node_id: str) -> ClusterNode | None:
    result = await session.execute(select(ClusterNode).where(ClusterNode.id == node_id))
    return result.scalar_one_or_none()


async def list_cluster_nodes(session: AsyncSession) -> list[ClusterNode]:
    result = await session.execute(select(ClusterNode).order_by(ClusterNode.created_at.desc()))
    return list(result.scalars().all())


async def delete_cluster_node(session: AsyncSession, node_id: str) -> bool:
    node = await get_cluster_node(session, node_id)
    if not node:
        return False
    await session.delete(node)
    await session.commit()
    logger.info("Deleted cluster node: id=%s", node_id)
    return True


# -- EvaluationResult CRUD --

async def create_evaluation(
    session: AsyncSession,
    *,
    model_id: str,
    benchmark_name: str,
    tenant_id: str = "",
    version_id: str = "",
) -> EvaluationResult:
    e = EvaluationResult(
        model_id=model_id, benchmark_name=benchmark_name,
        tenant_id=tenant_id, version_id=version_id,
    )
    session.add(e)
    await session.commit()
    await session.refresh(e)
    logger.info("Created evaluation: id=%s model=%s benchmark=%s", e.id, model_id, benchmark_name)
    return e


async def get_evaluation(session: AsyncSession, eval_id: str) -> EvaluationResult | None:
    result = await session.execute(select(EvaluationResult).where(EvaluationResult.id == eval_id))
    return result.scalar_one_or_none()


async def list_evaluations(
    session: AsyncSession,
    *,
    model_id: str = "",
    version_id: str = "",
    benchmark_name: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[EvaluationResult], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    query = select(EvaluationResult)
    count_query = select(func.count()).select_from(EvaluationResult)
    if model_id:
        query = query.where(EvaluationResult.model_id == model_id)
        count_query = count_query.where(EvaluationResult.model_id == model_id)
    if version_id:
        query = query.where(EvaluationResult.version_id == version_id)
        count_query = count_query.where(EvaluationResult.version_id == version_id)
    if benchmark_name:
        query = query.where(EvaluationResult.benchmark_name == benchmark_name)
        count_query = count_query.where(EvaluationResult.benchmark_name == benchmark_name)
    if status:
        query = query.where(EvaluationResult.status == status)
        count_query = count_query.where(EvaluationResult.status == status)
    total = (await session.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(EvaluationResult.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    return list(result.scalars().all()), total


_EVALUATION_UPDATABLE = {"status", "score", "metrics", "error_message", "completed_at"}


async def update_evaluation(session: AsyncSession, eval_id: str, **fields) -> EvaluationResult | None:
    e = await get_evaluation(session, eval_id)
    if not e:
        return None
    for k, val in fields.items():
        if k in _EVALUATION_UPDATABLE and val is not None:
            setattr(e, k, val)
    await session.commit()
    await session.refresh(e)
    logger.info("Updated evaluation: id=%s fields=%s", eval_id, list(fields.keys()))
    return e


async def delete_evaluation(session: AsyncSession, eval_id: str) -> bool:
    e = await get_evaluation(session, eval_id)
    if not e:
        return False
    await session.delete(e)
    await session.commit()
    logger.info("Deleted evaluation: id=%s", eval_id)
    return True


# SecurityScan CRUD

_SECURITY_SCAN_UPDATABLE = {"status", "findings", "risk_level", "completed_at"}


async def create_security_scan(
    session: AsyncSession, *, model_id: str, version_id: str = "",
    scan_type: str = "full",
) -> SecurityScan:
    scan = SecurityScan(model_id=model_id, version_id=version_id, scan_type=scan_type)
    session.add(scan)
    await session.commit()
    await session.refresh(scan)
    logger.info("Created security scan: id=%s model=%s", scan.id, model_id)
    return scan


async def get_security_scan(session: AsyncSession, scan_id: str) -> SecurityScan | None:
    result = await session.execute(select(SecurityScan).where(SecurityScan.id == scan_id))
    return result.scalar_one_or_none()


async def list_security_scans(
    session: AsyncSession, *, model_id: str = "", version_id: str = "",
    status: str = "", page: int = 1, page_size: int = 20,
) -> tuple[list[SecurityScan], int]:
    q = select(SecurityScan)
    c = select(func.count()).select_from(SecurityScan)
    if model_id:
        q = q.where(SecurityScan.model_id == model_id)
        c = c.where(SecurityScan.model_id == model_id)
    if version_id:
        q = q.where(SecurityScan.version_id == version_id)
        c = c.where(SecurityScan.version_id == version_id)
    if status:
        q = q.where(SecurityScan.status == status)
        c = c.where(SecurityScan.status == status)
    total = (await session.execute(c)).scalar() or 0
    q = q.order_by(SecurityScan.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(q)
    return list(result.scalars().all()), total


async def update_security_scan(session: AsyncSession, scan_id: str, **fields) -> SecurityScan | None:
    result = await session.execute(select(SecurityScan).where(SecurityScan.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        return None
    for k, v in fields.items():
        if k in _SECURITY_SCAN_UPDATABLE:
            setattr(scan, k, v)
    await session.commit()
    await session.refresh(scan)
    return scan


# Watermark CRUD

async def create_watermark(
    session: AsyncSession, *, model_id: str, version_id: str = "",
    watermark_type: str = "metadata", payload: str = "{}", signature: str = "",
) -> Watermark:
    wm = Watermark(
        model_id=model_id, version_id=version_id,
        watermark_type=watermark_type, payload=payload, signature=signature,
    )
    session.add(wm)
    await session.commit()
    await session.refresh(wm)
    logger.info("Created watermark: id=%s model=%s", wm.id, model_id)
    return wm


async def get_watermark(session: AsyncSession, wm_id: str) -> Watermark | None:
    result = await session.execute(select(Watermark).where(Watermark.id == wm_id))
    return result.scalar_one_or_none()


async def list_watermarks(
    session: AsyncSession, *, model_id: str = "", version_id: str = "",
) -> list[Watermark]:
    q = select(Watermark)
    if model_id:
        q = q.where(Watermark.model_id == model_id)
    if version_id:
        q = q.where(Watermark.version_id == version_id)
    result = await session.execute(q.order_by(Watermark.created_at.desc()))
    return list(result.scalars().all())


# ApprovalRequest CRUD

_APPROVAL_UPDATABLE = {"status", "approver", "comment", "updated_at"}


async def create_approval_request(
    session: AsyncSession, *, model_id: str, version_id: str = "",
    level: str = "l1", requester: str = "", tenant_id: str = "",
) -> ApprovalRequest:
    ar = ApprovalRequest(
        model_id=model_id, version_id=version_id,
        level=ApprovalLevel(level), requester=requester, tenant_id=tenant_id,
    )
    session.add(ar)
    await session.commit()
    await session.refresh(ar)
    logger.info("Created approval request: id=%s model=%s level=%s", ar.id, model_id, level)
    return ar


async def get_approval_request(session: AsyncSession, req_id: str) -> ApprovalRequest | None:
    result = await session.execute(select(ApprovalRequest).where(ApprovalRequest.id == req_id))
    return result.scalar_one_or_none()


async def list_approval_requests(
    session: AsyncSession, *, model_id: str = "", status: str = "",
    level: str = "", page: int = 1, page_size: int = 20,
) -> tuple[list[ApprovalRequest], int]:
    q = select(ApprovalRequest)
    c = select(func.count()).select_from(ApprovalRequest)
    if model_id:
        q = q.where(ApprovalRequest.model_id == model_id)
        c = c.where(ApprovalRequest.model_id == model_id)
    if status:
        q = q.where(ApprovalRequest.status == status)
        c = c.where(ApprovalRequest.status == status)
    if level:
        q = q.where(ApprovalRequest.level == level)
        c = c.where(ApprovalRequest.level == level)
    total = (await session.execute(c)).scalar() or 0
    q = q.order_by(ApprovalRequest.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(q)
    return list(result.scalars().all()), total


async def update_approval_request(session: AsyncSession, req_id: str, **fields) -> ApprovalRequest | None:
    result = await session.execute(select(ApprovalRequest).where(ApprovalRequest.id == req_id))
    ar = result.scalar_one_or_none()
    if not ar:
        return None
    for k, v in fields.items():
        if k in _APPROVAL_UPDATABLE:
            setattr(ar, k, v)
    await session.commit()
    await session.refresh(ar)
    return ar


# LoraMergeTask CRUD

_LORA_MERGE_UPDATABLE = {"status", "output_version_id", "error_message", "completed_at"}


async def create_lora_merge_task(
    session: AsyncSession, *, base_version_id: str, lora_version_id: str,
    target_format: str = "mlx", quant_bits: int = 4,
) -> LoraMergeTask:
    task = LoraMergeTask(
        base_version_id=base_version_id, lora_version_id=lora_version_id,
        target_format=target_format, quant_bits=quant_bits,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    logger.info("Created LoRA merge task: id=%s", task.id)
    return task


async def get_lora_merge_task(session: AsyncSession, task_id: str) -> LoraMergeTask | None:
    result = await session.execute(select(LoraMergeTask).where(LoraMergeTask.id == task_id))
    return result.scalar_one_or_none()


async def list_lora_merge_tasks(
    session: AsyncSession, *, status: str = "", page: int = 1, page_size: int = 20,
) -> tuple[list[LoraMergeTask], int]:
    q = select(LoraMergeTask)
    c = select(func.count()).select_from(LoraMergeTask)
    if status:
        q = q.where(LoraMergeTask.status == status)
        c = c.where(LoraMergeTask.status == status)
    total = (await session.execute(c)).scalar() or 0
    q = q.order_by(LoraMergeTask.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(q)
    return list(result.scalars().all()), total


async def update_lora_merge_task(session: AsyncSession, task_id: str, **fields) -> LoraMergeTask | None:
    result = await session.execute(select(LoraMergeTask).where(LoraMergeTask.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        return None
    for k, v in fields.items():
        if k in _LORA_MERGE_UPDATABLE:
            setattr(task, k, v)
    await session.commit()
    await session.refresh(task)
    return task


# DistributedTask CRUD

_DISTRIBUTED_TASK_UPDATABLE = {"status", "progress", "completed_at"}


async def create_distributed_task(
    session: AsyncSession, *, model_id: str, version_id: str = "",
    target_nodes: str = "[]",
) -> DistributedTask:
    task = DistributedTask(model_id=model_id, version_id=version_id, target_nodes=target_nodes)
    session.add(task)
    await session.commit()
    await session.refresh(task)
    logger.info("Created distributed task: id=%s model=%s", task.id, model_id)
    return task


async def get_distributed_task(session: AsyncSession, task_id: str) -> DistributedTask | None:
    result = await session.execute(select(DistributedTask).where(DistributedTask.id == task_id))
    return result.scalar_one_or_none()


async def update_distributed_task(session: AsyncSession, task_id: str, **fields) -> DistributedTask | None:
    result = await session.execute(select(DistributedTask).where(DistributedTask.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        return None
    for k, v in fields.items():
        if k in _DISTRIBUTED_TASK_UPDATABLE:
            setattr(task, k, v)
    await session.commit()
    await session.refresh(task)
    return task


# GitLfsLock CRUD

async def create_gitlfs_lock(
    session: AsyncSession, *, model_id: str, path: str, owner: str = "",
) -> GitLfsLock:
    lock = GitLfsLock(model_id=model_id, path=path, owner=owner)
    session.add(lock)
    await session.commit()
    await session.refresh(lock)
    logger.info("Created Git LFS lock: id=%s path=%s", lock.id, path)
    return lock


async def list_gitlfs_locks(session: AsyncSession, *, model_id: str = "", path: str = "") -> list[GitLfsLock]:
    q = select(GitLfsLock)
    if model_id:
        q = q.where(GitLfsLock.model_id == model_id)
    if path:
        q = q.where(GitLfsLock.path == path)
    result = await session.execute(q.order_by(GitLfsLock.created_at.desc()))
    return list(result.scalars().all())


async def delete_gitlfs_lock(session: AsyncSession, lock_id: str) -> bool:
    result = await session.execute(select(GitLfsLock).where(GitLfsLock.id == lock_id))
    lock = result.scalar_one_or_none()
    if not lock:
        return False
    await session.delete(lock)
    await session.commit()
    return True


# -- ModelRating CRUD --

async def create_model_rating(
    session: AsyncSession, *, model_id: str, user_id: str = "",
    score: int = 0, comment: str = "",
) -> ModelRating:
    r = ModelRating(model_id=model_id, user_id=user_id, score=score, comment=comment)
    session.add(r)
    await session.commit()
    await session.refresh(r)
    logger.info("Created model rating: id=%s model=%s score=%d", r.id, model_id, score)
    return r


async def get_model_rating(session: AsyncSession, rating_id: str) -> ModelRating | None:
    result = await session.execute(select(ModelRating).where(ModelRating.id == rating_id))
    return result.scalar_one_or_none()


async def list_model_ratings(
    session: AsyncSession, *, model_id: str = "", user_id: str = "",
    page: int = 1, page_size: int = 20,
) -> tuple[list[ModelRating], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    q = select(ModelRating)
    c = select(func.count()).select_from(ModelRating)
    if model_id:
        q = q.where(ModelRating.model_id == model_id)
        c = c.where(ModelRating.model_id == model_id)
    if user_id:
        q = q.where(ModelRating.user_id == user_id)
        c = c.where(ModelRating.user_id == user_id)
    total = (await session.execute(c)).scalar() or 0
    q = q.order_by(ModelRating.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(q)
    return list(result.scalars().all()), total


async def delete_model_rating(session: AsyncSession, rating_id: str) -> bool:
    r = await get_model_rating(session, rating_id)
    if not r:
        return False
    await session.delete(r)
    await session.commit()
    logger.info("Deleted model rating: id=%s", rating_id)
    return True


async def get_model_avg_rating(session: AsyncSession, model_id: str) -> float:
    result = await session.execute(
        select(func.avg(ModelRating.score)).where(ModelRating.model_id == model_id)
    )
    val = result.scalar()
    return float(val) if val else 0.0


# -- ModelFavorite CRUD --

async def create_model_favorite(
    session: AsyncSession, *, model_id: str, user_id: str = "",
) -> ModelFavorite:
    f = ModelFavorite(model_id=model_id, user_id=user_id)
    session.add(f)
    await session.commit()
    await session.refresh(f)
    logger.info("Created model favorite: id=%s model=%s user=%s", f.id, model_id, user_id)
    return f


async def list_model_favorites(
    session: AsyncSession, *, user_id: str = "", model_id: str = "",
    page: int = 1, page_size: int = 20,
) -> tuple[list[ModelFavorite], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    q = select(ModelFavorite)
    c = select(func.count()).select_from(ModelFavorite)
    if user_id:
        q = q.where(ModelFavorite.user_id == user_id)
        c = c.where(ModelFavorite.user_id == user_id)
    if model_id:
        q = q.where(ModelFavorite.model_id == model_id)
        c = c.where(ModelFavorite.model_id == model_id)
    total = (await session.execute(c)).scalar() or 0
    q = q.order_by(ModelFavorite.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(q)
    return list(result.scalars().all()), total


async def delete_model_favorite(session: AsyncSession, favorite_id: str) -> bool:
    result = await session.execute(select(ModelFavorite).where(ModelFavorite.id == favorite_id))
    f = result.scalar_one_or_none()
    if not f:
        return False
    await session.delete(f)
    await session.commit()
    logger.info("Deleted model favorite: id=%s", favorite_id)
    return True


async def is_model_favorited(session: AsyncSession, model_id: str, user_id: str) -> bool:
    result = await session.execute(
        select(ModelFavorite).where(
            ModelFavorite.model_id == model_id, ModelFavorite.user_id == user_id,
        )
    )
    return result.scalar_one_or_none() is not None


# -- ModelBranch CRUD --

_BRANCH_UPDATABLE = {"head_version_id", "status", "description"}


async def create_model_branch(
    session: AsyncSession, *, model_id: str, name: str,
    base_version_id: str = "", description: str = "",
) -> ModelBranch:
    b = ModelBranch(
        model_id=model_id, name=name,
        base_version_id=base_version_id, description=description,
    )
    session.add(b)
    await session.commit()
    await session.refresh(b)
    logger.info("Created model branch: id=%s model=%s name=%s", b.id, model_id, name)
    return b


async def get_model_branch(session: AsyncSession, branch_id: str) -> ModelBranch | None:
    result = await session.execute(select(ModelBranch).where(ModelBranch.id == branch_id))
    return result.scalar_one_or_none()


async def list_model_branches(
    session: AsyncSession, *, model_id: str = "", status: str = "",
) -> list[ModelBranch]:
    q = select(ModelBranch)
    if model_id:
        q = q.where(ModelBranch.model_id == model_id)
    if status:
        q = q.where(ModelBranch.status == BranchStatus(status))
    result = await session.execute(q.order_by(ModelBranch.created_at.desc()))
    return list(result.scalars().all())


async def update_model_branch(session: AsyncSession, branch_id: str, **fields) -> ModelBranch | None:
    b = await get_model_branch(session, branch_id)
    if not b:
        return None
    for k, v in fields.items():
        if k in _BRANCH_UPDATABLE and v is not None:
            setattr(b, k, v)
    await session.commit()
    await session.refresh(b)
    logger.info("Updated model branch: id=%s fields=%s", branch_id, list(fields.keys()))
    return b


async def delete_model_branch(session: AsyncSession, branch_id: str) -> bool:
    b = await get_model_branch(session, branch_id)
    if not b:
        return False
    await session.delete(b)
    await session.commit()
    logger.info("Deleted model branch: id=%s", branch_id)
    return True


# -- DownloadTask CRUD --

async def create_download_task(
    session: AsyncSession,
    *,
    model_id: str,
    source_url: str,
    version_id: str = "",
    speed_limit_kbps: int = 0,
    max_retries: int = 3,
) -> DownloadTask:
    t = DownloadTask(
        model_id=model_id, source_url=source_url,
        version_id=version_id, speed_limit_kbps=speed_limit_kbps,
        max_retries=max_retries,
    )
    session.add(t)
    await session.commit()
    await session.refresh(t)
    logger.info("Created download task: id=%s model=%s url=%s", t.id, model_id, source_url)
    return t


async def get_download_task(session: AsyncSession, task_id: str) -> DownloadTask | None:
    result = await session.execute(select(DownloadTask).where(DownloadTask.id == task_id))
    return result.scalar_one_or_none()


async def list_download_tasks(
    session: AsyncSession,
    *,
    model_id: str = "",
    status: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[DownloadTask], int]:
    page_size = min(page_size, MAX_PAGE_SIZE)
    query = select(DownloadTask)
    count_query = select(func.count()).select_from(DownloadTask)
    if model_id:
        query = query.where(DownloadTask.model_id == model_id)
        count_query = count_query.where(DownloadTask.model_id == model_id)
    if status:
        query = query.where(DownloadTask.status == status)
        count_query = count_query.where(DownloadTask.status == status)
    total = (await session.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(DownloadTask.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    return list(result.scalars().all()), total


_DOWNLOAD_TASK_UPDATABLE = {
    "status", "progress_percent", "downloaded_bytes", "total_bytes",
    "retry_count", "error_message", "file_path",
}


async def update_download_task(
    session: AsyncSession,
    task_id: str,
    **fields,
) -> DownloadTask | None:
    t = await get_download_task(session, task_id)
    if not t:
        return None
    for k, val in fields.items():
        if k in _DOWNLOAD_TASK_UPDATABLE and val is not None:
            setattr(t, k, val)
    await session.commit()
    await session.refresh(t)
    logger.info("Updated download task: id=%s fields=%s", task_id, list(fields.keys()))
    return t
