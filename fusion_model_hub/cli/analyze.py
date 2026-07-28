from __future__ import annotations

import asyncio
import json
import logging

import typer

logger = logging.getLogger(__name__)

analyze_app = typer.Typer(no_args_is_help=True)


@analyze_app.command("assess")
def analyze_assess(
    model_id: str = typer.Argument(..., help="Model ID to assess"),
):
    result = asyncio.run(_assess(model_id))
    typer.echo(json.dumps(result, indent=2))


async def _assess(model_id: str) -> dict:
    from ..adapt.decision import AdaptDecisionEngine
    from ..server.config import Settings

    settings = Settings()
    engine = AdaptDecisionEngine(settings.mlx_url)
    result = await engine.assess(model_id)
    logger.info("Assessment for %s: level=%s", model_id, result.level.value)
    return result.model_dump()


@analyze_app.command("plan")
def analyze_plan(
    model_id: str = typer.Argument(..., help="Model ID to plan migration"),
):
    result = asyncio.run(_plan(model_id))
    typer.echo(json.dumps(result, indent=2))


async def _plan(model_id: str) -> dict:
    from ..adapt.decision import AdaptDecisionEngine
    from ..server.config import Settings

    settings = Settings()
    engine = AdaptDecisionEngine(settings.mlx_url)
    plan = await engine.assess_and_plan(model_id)
    logger.info("Migration plan for %s: level=%s steps=%d", model_id, plan.level.value, len(plan.steps))
    return plan.model_dump()
