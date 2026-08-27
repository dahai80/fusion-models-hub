from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_model_hub.db.database import get_engine, init_db
from fusion_model_hub.db.models import EvaluationStatus
from fusion_model_hub.server.config import Settings
from fusion_model_hub.server.deps import get_session_factory, init_deps
from fusion_model_hub.server.eval_tasks import _run_evaluation, submit_evaluation


def _mock_response(json_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def _mock_async_client():
    # httpx.AsyncClient is used as `async with AsyncClient(...) as client`.
    # AsyncMock.__aenter__ returns a fresh MagicMock by default — wire it to
    # return the same mock so `client.post` resolves to the configured calls.
    c = AsyncMock()
    c.__aenter__.return_value = c
    c.__aexit__.return_value = None
    return c


@pytest.fixture
async def runner_env():
    settings = Settings(
        host="127.0.0.1",
        port=11444,
        data_dir="/tmp/fmh_test_eval_runner",
        db_url="sqlite+aiosqlite:///:memory:",
        log_level="WARNING",
        eval_runner_enabled=True,
        bench_url="http://bench.test:8090",
        bench_api_key="bench-key",
    )
    engine = get_engine(settings.db_url)
    await init_db(engine)
    init_deps(settings, engine)
    yield settings
    # Drain any spawned runner tasks so they do not leak across tests.
    import fusion_model_hub.server.eval_tasks as et

    for t in list(et._running_evals.values()):
        t.cancel()


async def _seed_eval(settings, model_id="m-eval-1"):
    from fusion_model_hub.db import crud

    sf = get_session_factory()
    async with sf() as session:
        # Need a real Model row for _resolve_model_name (hf_repo or name).
        m = await crud.create_model(
            session,
            name="eval-test-model",
            hf_repo="mlx-community/eval-test-4bit",
            model_type="llm",
        )
        e = await crud.create_evaluation(
            session,
            model_id=m.id,
            benchmark_name="mmlu",
        )
        return e.id, m.id


class TestEvaluationRunner:
    @pytest.mark.asyncio
    async def test_run_evaluation_completes_with_score(self, runner_env):
        eval_id, model_id = await _seed_eval(runner_env)
        # Mock httpx.AsyncClient: POST /tasks -> task_id; first GET /tasks/{id}
        # -> completed; GET /results/{id} -> metric_value 0.82.
        mock_client = _mock_async_client()
        mock_client.post.return_value = _mock_response({"task_id": "bench-task-1"}, 201)
        mock_client.get.side_effect = [
            _mock_response({"status": "completed", "task_id": "bench-task-1"}, 200),
            _mock_response(
                {"task_id": "bench-task-1", "metric_value": 0.82, "metric_name": "accuracy", "pass_rate": 0.82},
                200,
            ),
        ]
        with (
            patch("fusion_model_hub.server.eval_tasks.httpx.AsyncClient", return_value=mock_client),
            patch("fusion_model_hub.server.eval_tasks.asyncio.sleep", new=AsyncMock()),
        ):
            await _run_evaluation(eval_id, model_id, "", "mmlu", runner_env)

        # Row should be COMPLETED with score 0.82 + metrics containing bench_task_id.
        from fusion_model_hub.db import crud

        sf = get_session_factory()
        async with sf() as session:
            e = await crud.get_evaluation(session, eval_id)
            assert e is not None
            assert e.status == EvaluationStatus.COMPLETED
            assert abs(e.score - 0.82) < 1e-6
            assert "bench-task-1" in e.metrics
            assert e.error_message == ""

    @pytest.mark.asyncio
    async def test_run_evaluation_bench_rejects_task(self, runner_env):
        eval_id, model_id = await _seed_eval(runner_env)
        mock_client = _mock_async_client()
        mock_client.post.return_value = _mock_response({"detail": "forbidden"}, 403)
        with (
            patch("fusion_model_hub.server.eval_tasks.httpx.AsyncClient", return_value=mock_client),
            patch("fusion_model_hub.server.eval_tasks.asyncio.sleep", new=AsyncMock()),
        ):
            await _run_evaluation(eval_id, model_id, "", "mmlu", runner_env)

        from fusion_model_hub.db import crud

        sf = get_session_factory()
        async with sf() as session:
            e = await crud.get_evaluation(session, eval_id)
            assert e is not None
            assert e.status == EvaluationStatus.FAILED
            assert "403" in e.error_message

    @pytest.mark.asyncio
    async def test_run_evaluation_bench_unreachable(self, runner_env):
        import httpx

        eval_id, model_id = await _seed_eval(runner_env)
        mock_client = _mock_async_client()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        with (
            patch("fusion_model_hub.server.eval_tasks.httpx.AsyncClient", return_value=mock_client),
            patch("fusion_model_hub.server.eval_tasks.asyncio.sleep", new=AsyncMock()),
        ):
            await _run_evaluation(eval_id, model_id, "", "mmlu", runner_env)

        from fusion_model_hub.db import crud

        sf = get_session_factory()
        async with sf() as session:
            e = await crud.get_evaluation(session, eval_id)
            assert e is not None
            assert e.status == EvaluationStatus.FAILED
            assert "not available" in e.error_message

    @pytest.mark.asyncio
    async def test_run_evaluation_bench_task_failed(self, runner_env):
        eval_id, model_id = await _seed_eval(runner_env)
        mock_client = _mock_async_client()
        mock_client.post.return_value = _mock_response({"task_id": "bench-task-2"}, 201)
        mock_client.get.return_value = _mock_response({"status": "failed"}, 200)
        with (
            patch("fusion_model_hub.server.eval_tasks.httpx.AsyncClient", return_value=mock_client),
            patch("fusion_model_hub.server.eval_tasks.asyncio.sleep", new=AsyncMock()),
        ):
            await _run_evaluation(eval_id, model_id, "", "mmlu", runner_env)

        from fusion_model_hub.db import crud

        sf = get_session_factory()
        async with sf() as session:
            e = await crud.get_evaluation(session, eval_id)
            assert e is not None
            assert e.status == EvaluationStatus.FAILED
            assert "bench-task-2" in e.error_message

    @pytest.mark.asyncio
    async def test_run_evaluation_no_bench_url(self):
        # bench_url empty -> immediate FAILED without touching httpx.
        settings = Settings(
            db_url="sqlite+aiosqlite:///:memory:",
            data_dir="/tmp/fmh_test_eval_nourl",
            log_level="WARNING",
            eval_runner_enabled=True,
            bench_url="",
        )
        # Force bench_url empty (Settings.__post_init__ would default it from env).
        settings.bench_url = ""
        engine = get_engine(settings.db_url)
        await init_db(engine)
        init_deps(settings, engine)
        from fusion_model_hub.db import crud

        sf = get_session_factory()
        async with sf() as session:
            m = await crud.create_model(session, name="no-url-model", model_type="llm")
            e = await crud.create_evaluation(session, model_id=m.id, benchmark_name="mmlu")
            eval_id, model_id = e.id, m.id

        with patch("fusion_model_hub.server.eval_tasks.httpx.AsyncClient") as MockC:
            await _run_evaluation(eval_id, model_id, "", "mmlu", settings)
            MockC.assert_not_called()

        async with sf() as session:
            e = await crud.get_evaluation(session, eval_id)
            assert e.status == EvaluationStatus.FAILED
            assert "not configured" in e.error_message

    @pytest.mark.asyncio
    async def test_submit_evaluation_disabled_noop(self, runner_env):
        runner_env.eval_runner_enabled = False
        eval_id, model_id = await _seed_eval(runner_env)
        with patch("fusion_model_hub.server.eval_tasks._run_evaluation") as mock_run:
            await submit_evaluation(eval_id, model_id, "", "mmlu")
            mock_run.assert_not_called()
        # No background task registered.
        import fusion_model_hub.server.eval_tasks as et

        assert eval_id not in et._running_evals

    @pytest.mark.asyncio
    async def test_submit_evaluation_enabled_spawns_task(self, runner_env):
        eval_id, model_id = await _seed_eval(runner_env)
        mock_client = _mock_async_client()
        mock_client.post.return_value = _mock_response({"task_id": "bench-task-3"}, 201)
        mock_client.get.side_effect = [
            _mock_response({"status": "completed"}, 200),
            _mock_response({"metric_value": 0.9}, 200),
        ]
        with (
            patch("fusion_model_hub.server.eval_tasks.httpx.AsyncClient", return_value=mock_client),
            patch("fusion_model_hub.server.eval_tasks.asyncio.sleep", new=AsyncMock()),
        ):
            await submit_evaluation(eval_id, model_id, "", "mmlu")
            # Let the spawned task run to completion.
            t = None
            import fusion_model_hub.server.eval_tasks as et

            t = et._running_evals.get(eval_id)
            if t:
                await t

        from fusion_model_hub.db import crud

        sf = get_session_factory()
        async with sf() as session:
            e = await crud.get_evaluation(session, eval_id)
            assert e.status == EvaluationStatus.COMPLETED
            assert abs(e.score - 0.9) < 1e-6
