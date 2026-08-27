from __future__ import annotations

import asyncio
import json
import logging

import typer

logger = logging.getLogger(__name__)

recommend_app = typer.Typer(no_args_is_help=True)


@recommend_app.command("models")
def recommend_models(
    task: str = typer.Option("llm", "--task", "-t", help="Task type: llm|text2image|text2video|embedding"),
    preference: str = typer.Option("balanced", "--preference", "-p", help="Preference: quality|balanced|speed"),
    max_results: int = typer.Option(10, "--max-results", "-n", help="Max results"),
    min_params: float = typer.Option(0, "--min-params", help="Min model size in billions"),
    max_params: float = typer.Option(1000, "--max-params", help="Max model size in billions"),
):
    result = asyncio.run(_recommend(task, preference, max_results, min_params, max_params))
    typer.echo(json.dumps(result, indent=2))


async def _recommend(task: str, preference: str, max_results: int, min_params: float, max_params: float) -> dict:
    from ..recommend.engine import RecommendEngine
    from ..recommend.types import RecommendRequest
    from ..server.config import Settings

    settings = Settings()
    engine = RecommendEngine(settings.mlx_url)
    request = RecommendRequest(
        task=task,
        preference=preference,
        max_results=max_results,
        min_params_b=min_params,
        max_params_b=max_params,
    )

    models_from_db = await _fetch_models_from_api(settings)
    response = await engine.recommend(request, models_from_db)
    logger.info("Recommended %d models out of %d evaluated", len(response.recommendations), response.total_evaluated)
    return response.model_dump()


@recommend_app.command("quick")
def recommend_quick(
    task: str = typer.Option("llm", "--task", "-t", help="Task type"),
    preference: str = typer.Option("balanced", "--preference", "-p", help="Preference"),
):
    result = asyncio.run(_recommend(task, preference, 5, 0, 1000))
    if "recommendations" in result:
        for rec in result["recommendations"]:
            status = "✓" if rec.get("can_run") else "✗"
            typer.echo(f"  {status} {rec['name']:40s} score={rec.get('rank_score', 0):.1f}  {rec.get('reason', '')}")
    else:
        typer.echo(json.dumps(result, indent=2))


async def _fetch_models_from_api(settings) -> list[dict]:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            base_url = (
                f"http://{settings.host}:{settings.port}" if hasattr(settings, "host") else "http://localhost:11444"
            )
            resp = await client.get(f"{base_url}/api/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("items", [])
    except Exception as e:
        logger.warning("Could not fetch models from API: %s", e)
    return []
