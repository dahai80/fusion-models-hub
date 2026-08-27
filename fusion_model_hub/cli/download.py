from __future__ import annotations

import asyncio
import logging

import typer

logger = logging.getLogger(__name__)

download_app = typer.Typer(no_args_is_help=True)


@download_app.command("hf")
def download_hf(
    model_id: str = typer.Argument(..., help="HuggingFace model ID, e.g. mlx-community/Llama-3.2-1B-Instruct-4bit"),
    mirror: str = typer.Option("https://hf-mirror.com", "--mirror", help="HF mirror URL"),
    storage_dir: str = typer.Option("", "--storage-dir", help="Local storage directory"),
    expected_hash: str = typer.Option("", "--hash", help="Expected SHA256 hash for verification"),
):
    import json

    result = asyncio.run(_download_hf(model_id, mirror, storage_dir, expected_hash))
    typer.echo(json.dumps(result, indent=2))


async def _download_hf(model_id: str, mirror: str, storage_dir: str, expected_hash: str) -> dict:
    from ..repo.downloader import ModelDownloader

    url = f"{mirror}/{model_id}"
    downloader = ModelDownloader(storage_dir)
    result = await downloader.download(url, model_id.replace("/", "--"), expected_hash)
    logger.info("Download result for %s: %s", model_id, result.get("status"))
    return result


@download_app.command("url")
def download_url(
    url: str = typer.Argument(..., help="Direct download URL"),
    model_id: str = typer.Argument(..., help="Local model identifier"),
    storage_dir: str = typer.Option("", "--storage-dir", help="Local storage directory"),
    expected_hash: str = typer.Option("", "--hash", help="Expected SHA256 hash for verification"),
):
    import json

    result = asyncio.run(_download_url(url, model_id, storage_dir, expected_hash))
    typer.echo(json.dumps(result, indent=2))


async def _download_url(url: str, model_id: str, storage_dir: str, expected_hash: str) -> dict:
    from ..repo.downloader import ModelDownloader

    downloader = ModelDownloader(storage_dir)
    result = await downloader.download(url, model_id, expected_hash)
    logger.info("Download result for %s: %s", model_id, result.get("status"))
    return result
