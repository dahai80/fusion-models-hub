import asyncio
import json
import logging
from datetime import UTC, datetime

import httpx

from ..db import crud
from ..db.models import EvaluationStatus
from .config import Settings
from .deps import get_session_factory, get_settings

logger = logging.getLogger(__name__)

_running_evals: dict[str, asyncio.Task] = {}

# R1: bound concurrent evaluations. Each eval drives a Fusion-Bench task
# that loads + runs a model (GBs of unified memory + minutes of compute).
# Unbounded create_task lets 50 concurrent POSTs fan out 50 bench tasks and
# saturate the bench/MLX node. The DB row is created PENDING immediately so
# the client can poll; the worker flips it to RUNNING when it acquires the slot.
_EVAL_CONCURRENCY = 2
_eval_semaphore: asyncio.Semaphore | None = None


def _get_eval_semaphore() -> asyncio.Semaphore:
    global _eval_semaphore
    if _eval_semaphore is None:
        _eval_semaphore = asyncio.Semaphore(_EVAL_CONCURRENCY)
    return _eval_semaphore


def _bench_headers(settings: Settings) -> dict[str, str]:
    # Fusion-Bench IdentityMiddleware reads x-api-key; an empty key falls
    # through to anonymous VIEWER (no TASK_CREATE) and bench returns 403.
    if settings.bench_api_key:
        return {"X-API-Key": settings.bench_api_key}
    return {}


# Fusion-Bench TaskCreateRequest._SUITE_MAP maps quick/standard/full→speed.
# benchmark_name may be a custom suite/executor; pass it as `suite` so the
# bench resolves the executor, and fall back to the `speed` executor key.
_SUITE_TO_EXECUTOR = {
    "quick": "speed",
    "standard": "speed",
    "full": "speed",
}


async def _resolve_model_name(model_id: str, version_id: str) -> str:
    # Fusion-Bench loads a model on MLX by its repo/name; the Hub resolves the
    # same way inference does (m.hf_repo or m.name). A pinned version does not
    # change the MLX model name — it only selects weights — so the model-level
    # name is the correct load target.
    sf = get_session_factory()
    async with sf() as session:
        m = await crud.get_model(session, model_id)
        if not m:
            return model_id
        return m.hf_repo or m.name


async def submit_evaluation(
    eval_id: str,
    model_id: str,
    version_id: str,
    benchmark_name: str,
) -> str:
    settings = get_settings()
    if not getattr(settings, "eval_runner_enabled", True):
        logger.info("Evaluation runner disabled (eval_runner_enabled=False): id=%s stays PENDING", eval_id)
        return eval_id
    atask = asyncio.create_task(
        _run_evaluation(eval_id, model_id, version_id, benchmark_name, settings),
        name=f"eval-{eval_id}",
    )
    _running_evals[eval_id] = atask
    atask.add_done_callback(lambda t: _running_evals.pop(eval_id, None))
    logger.info("Submitted evaluation: id=%s model=%s benchmark=%s", eval_id, model_id, benchmark_name)
    return eval_id


async def resume_evaluation(
    eval_id: str,
    model_id: str,
    version_id: str,
    benchmark_name: str,
) -> str | None:
    settings = get_settings()
    atask = asyncio.create_task(
        _run_evaluation(eval_id, model_id, version_id, benchmark_name, settings),
        name=f"eval-{eval_id}",
    )
    _running_evals[eval_id] = atask
    atask.add_done_callback(lambda t: _running_evals.pop(eval_id, None))
    logger.info("Resumed orphaned evaluation: id=%s", eval_id)
    return eval_id


async def _run_evaluation(
    eval_id: str,
    model_id: str,
    version_id: str,
    benchmark_name: str,
    settings: Settings,
) -> None:
    async with _get_eval_semaphore():
        sf = get_session_factory()
        bench_url = settings.bench_url.rstrip("/")
        headers = _bench_headers(settings)
        try:
            async with sf() as session:
                await crud.update_evaluation(session, eval_id, status=EvaluationStatus.RUNNING)
        except Exception:
            logger.exception("Failed to flip eval to RUNNING: id=%s", eval_id)

        if not settings.bench_url:
            await _fail_eval(sf, eval_id, "Fusion-Bench URL not configured (FMH_BENCH_URL)")
            return

        model_name = await _resolve_model_name(model_id, version_id)
        executor_key = _SUITE_TO_EXECUTOR.get(benchmark_name, "speed")
        payload = {
            "model": model_name,
            "model_id": model_id,
            "suite": benchmark_name,
            "executor_key": executor_key,
            "level": "L1",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(f"{bench_url}/api/v1/tasks", json=payload, headers=headers)
                if resp.status_code not in (200, 201, 202):
                    body = resp.text[:300]
                    await _fail_eval(sf, eval_id, f"Fusion-Bench rejected task: {resp.status_code} {body}")
                    return
                bench_task_id = resp.json().get("task_id", "")
                if not bench_task_id:
                    await _fail_eval(sf, eval_id, "Fusion-Bench returned no task_id")
                    return
                logger.info("Eval %s → bench task %s", eval_id, bench_task_id)

                bench_status = await _poll_bench_task(client, bench_url, bench_task_id, headers)
                if bench_status not in ("completed", "skipped"):
                    err = f"Fusion-Bench task {bench_task_id} ended {bench_status}"
                    await _fail_eval(sf, eval_id, err)
                    return

                # completed/skipped → fetch result.
                score, metrics = await _fetch_bench_result(client, bench_url, bench_task_id, headers)
                async with sf() as session:
                    await crud.update_evaluation(
                        session,
                        eval_id,
                        status=EvaluationStatus.COMPLETED,
                        score=score,
                        metrics=metrics,
                        completed_at=datetime.now(UTC),
                    )
                logger.info("Evaluation completed: id=%s bench_task=%s score=%s", eval_id, bench_task_id, score)
        except httpx.ConnectError:
            logger.error("Fusion-Bench not available at %s for eval %s", bench_url, eval_id)
            await _fail_eval(sf, eval_id, f"Fusion-Bench not available at {bench_url}")
        except Exception:
            logger.exception("Evaluation runner failed: id=%s", eval_id)
            await _fail_eval(sf, eval_id, "Evaluation runner crashed — see hub logs")


async def _poll_bench_task(
    client: httpx.AsyncClient,
    bench_url: str,
    bench_task_id: str,
    headers: dict[str, str],
) -> str:
    # Poll up to ~10 min (120 x 5s). A bench task loads a model + runs a suite,
    # which can take minutes; 5s cadence balances responsiveness vs bench load.
    for _ in range(120):
        await asyncio.sleep(5)
        try:
            resp = await client.get(f"{bench_url}/api/v1/tasks/{bench_task_id}", headers=headers)
            if resp.status_code == 404:
                return "failed"
            if resp.status_code != 200:
                logger.warning("Bench task poll %s returned %d", bench_task_id, resp.status_code)
                continue
            st = resp.json().get("status", "")
            if st in ("completed", "failed", "skipped", "cancelled"):
                return st
        except httpx.ConnectError:
            logger.warning("Bench unreachable while polling task %s", bench_task_id)
        except Exception:
            logger.exception("Bench task poll error: %s", bench_task_id)
    return "timeout"


async def _fetch_bench_result(
    client: httpx.AsyncClient,
    bench_url: str,
    bench_task_id: str,
    headers: dict[str, str],
) -> tuple[float, str]:
    try:
        resp = await client.get(f"{bench_url}/api/v1/results/{bench_task_id}", headers=headers)
        if resp.status_code != 200:
            logger.warning("Bench result %s returned %d", bench_task_id, resp.status_code)
            return 0.0, json.dumps({"bench_task_id": bench_task_id, "error": f"result status {resp.status_code}"})
        data = resp.json()
        score = float(data.get("metric_value", 0.0) or 0.0)
        return score, json.dumps({"bench_task_id": bench_task_id, "result": data})
    except Exception:
        logger.exception("Failed to fetch bench result: %s", bench_task_id)
        return 0.0, json.dumps({"bench_task_id": bench_task_id, "error": "result fetch failed"})


async def _fail_eval(sf, eval_id: str, message: str) -> None:
    try:
        async with sf() as session:
            await crud.update_evaluation(
                session,
                eval_id,
                status=EvaluationStatus.FAILED,
                error_message=message,
                completed_at=datetime.now(UTC),
            )
    except Exception:
        logger.exception("Failed to mark eval FAILED: id=%s", eval_id)


def list_running_evals() -> list[dict]:
    return [{"id": eid, "name": t.get_name(), "done": t.done()} for eid, t in _running_evals.items()]
