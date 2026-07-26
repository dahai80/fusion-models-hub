import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ApiKey, AuditLog, ClusterNode, Model, ModelFormat, ModelTag, ModelType, ModelVersion, Quantization, QuantizeTask, TaskStatus, VersionStatus

logger = logging.getLogger(__name__)


async def create_model(
    session: AsyncSession,
    *,
    name: str,
    description: str = "",
    model_type: ModelType = ModelType.LLM,
    architecture: str = "",
    params_size: str = "",
    license: str = "",
    author: str = "",
    language: str = "",
    task_types: str = "",
    owner: str = "",
    hf_repo: str = "",
) -> Model:
    m = Model(
        name=name, description=description, model_type=model_type,
        architecture=architecture, params_size=params_size,
        license=license, author=author, language=language,
        task_types=task_types, owner=owner, hf_repo=hf_repo,
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
    keyword: str = "",
    model_type: str = "",
    architecture: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Model], int]:
    query = select(Model)
    count_query = select(func.count()).select_from(Model)

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


_MODEL_UPDATABLE = {"description", "model_type", "architecture", "params_size", "license", "author", "language", "task_types", "owner", "hf_repo"}


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
    status: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[ModelVersion], int]:
    query = select(ModelVersion).where(ModelVersion.model_id == model_id)
    count_query = select(func.count()).select_from(ModelVersion).where(ModelVersion.model_id == model_id)

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


class InvalidTransition(Exception):
    pass


async def update_version_status(
    session: AsyncSession, version_id: str, target_status: VersionStatus,
) -> ModelVersion | None:
    v = await get_version(session, version_id)
    if not v:
        return None
    allowed = VALID_TRANSITIONS.get(v.status, set())
    if target_status not in allowed:
        raise InvalidTransition(
            f"Cannot transition from {v.status.value} to {target_status.value}"
        )
    v.status = target_status
    await session.commit()
    await session.refresh(v)
    logger.info("Version status changed: id=%s -> %s", version_id, target_status.value)
    return v


_VERSION_UPDATABLE = {
    "file_path", "file_hash", "file_size", "release_notes",
    "benchmark_score", "inference_latency", "throughput", "memory_usage",
    "context_length", "successor_version_id",
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
) -> QuantizeTask:
    t = QuantizeTask(
        source_version_id=source_version_id,
        target_format=target_format,
        quant_bits=quant_bits,
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
) -> tuple[list[QuantizeTask], int]:
    query = select(QuantizeTask)
    count_query = select(func.count()).select_from(QuantizeTask)
    if status:
        query = query.where(QuantizeTask.status == status)
        count_query = count_query.where(QuantizeTask.status == status)
    total = (await session.execute(count_query)).scalar() or 0
    offset = (page - 1) * page_size
    query = query.order_by(QuantizeTask.created_at.desc()).offset(offset).limit(page_size)
    result = await session.execute(query)
    return list(result.scalars().all()), total


_TASK_UPDATABLE = {"status", "output_version_id", "error_message", "started_at", "completed_at"}


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
    permissions: str = "read,write",
) -> tuple[ApiKey, str]:
    full_key, key_hash, key_prefix = _generate_api_key()
    ak = ApiKey(name=name, key_hash=key_hash, key_prefix=key_prefix, permissions=permissions)
    session.add(ak)
    await session.commit()
    await session.refresh(ak)
    logger.info("Created API key: id=%s name=%s", ak.id, name)
    return ak, full_key


async def get_api_key(session: AsyncSession, key_id: str) -> ApiKey | None:
    result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
    return result.scalar_one_or_none()


async def list_api_keys(session: AsyncSession) -> list[ApiKey]:
    result = await session.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
    return list(result.scalars().all())


async def verify_api_key(session: AsyncSession, raw_key: str) -> ApiKey | None:
    key_hash = _hash_key(raw_key)
    result = await session.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
    )
    ak = result.scalar_one_or_none()
    if ak:
        ak.last_used_at = datetime.now(timezone.utc)
        await session.commit()
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
    detail: str = "",
) -> AuditLog:
    log = AuditLog(
        action=action, resource_type=resource_type,
        resource_id=resource_id, api_key_id=api_key_id,
        detail=detail,
    )
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log


async def list_audit_logs(
    session: AsyncSession,
    *,
    resource_type: str = "",
    action: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[AuditLog], int]:
    query = select(AuditLog)
    count_query = select(func.count()).select_from(AuditLog)
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
