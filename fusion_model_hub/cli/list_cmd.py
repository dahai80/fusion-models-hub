from __future__ import annotations

import asyncio
import json
import logging

import typer

logger = logging.getLogger(__name__)

list_app = typer.Typer(no_args_is_help=True)


@list_app.command("local")
def list_local(
    storage_dir: str = typer.Option("", "--storage-dir", help="Local models directory"),
):
    from ..manage.manager import LocalModelManager

    manager = LocalModelManager(storage_dir)
    models = manager.list()
    if not models:
        typer.echo("No local models found.")
        return
    for m in models:
        active = "*" if m.get("active") else " "
        size_info = ""
        p = m.get("path", "")
        if p:
            try:
                from pathlib import Path
                size_info = f" ({Path(p).stat().st_size / 1e9:.1f} GB)"
            except Exception:
                pass
        typer.echo(f"  {active} {m['id']:40s} {m.get('name', ''):30s}{size_info}")


@list_app.command("remote")
def list_remote(
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
):
    result = asyncio.run(_list_remote(limit))
    if isinstance(result, list):
        for m in result[:limit]:
            model_id = m.get("id", m.get("model_id", "unknown"))
            name = m.get("name", model_id)
            params = m.get("params_b", 0)
            typer.echo(f"  {name:40s} {params:.1f}B")
    else:
        typer.echo(json.dumps(result, indent=2))


async def _list_remote(limit: int) -> list[dict]:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:8080/api/v1/models", params={"limit": limit})
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("items", [])
    except Exception as e:
        logger.warning("Could not fetch remote models: %s", e)
    return []


@list_app.command("stats")
def list_stats(
    storage_dir: str = typer.Option("", "--storage-dir", help="Local models directory"),
):
    from ..manage.manager import LocalModelManager

    manager = LocalModelManager(storage_dir)
    stats = manager.get_stats()
    typer.echo(f"Total models: {stats['total_models']}")
    typer.echo(f"Active models: {stats['active_models']}")
    typer.echo(f"Total size: {stats['total_size_gb']:.2f} GB")
