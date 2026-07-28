from __future__ import annotations

import logging

import typer

from .analyze import analyze_app
from .download import download_app
from .list_cmd import list_app
from .recommend import recommend_app

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="fmh",
    help="Fusion Model Hub CLI — model discovery, download, recommendation, and analysis",
    no_args_is_help=True,
)

app.add_typer(download_app, name="download", help="Download models from HuggingFace or mirrors")
app.add_typer(recommend_app, name="recommend", help="Get model recommendations based on hardware")
app.add_typer(list_app, name="list", help="List local and remote models")
app.add_typer(analyze_app, name="analyze", help="Analyze model compatibility and adaptation")


@app.command()
def hardware():
    import asyncio


    result = asyncio.run(_show_hardware())
    if result:
        typer.echo(result)


async def _show_hardware() -> str:
    from ..hardware.detector import HardwareDetector
    from ..server.config import Settings

    settings = Settings()
    detector = HardwareDetector(settings.mlx_url)
    profile = await detector.detect()

    lines = ["=== Hardware Profile ==="]
    if profile.gpu:
        lines.append(f"GPU: {profile.gpu.name}")
        lines.append(f"  Chip: {profile.gpu.chip_generation.value}")
        lines.append(f"  VRAM: {profile.gpu.vram_gb:.1f} GB")
        lines.append(f"  Bandwidth: {profile.gpu.memory_bandwidth_gbps:.0f} GB/s")
    else:
        lines.append("GPU: Not detected")
    lines.append(f"CPU: {profile.cpu.name} ({profile.cpu.cores} cores)")
    lines.append(f"RAM: {profile.ram_gb:.1f} GB")
    lines.append(f"Disk Free: {profile.disk_free_gb:.1f} GB")
    lines.append(f"Effective VRAM: {profile.effective_vram_gb:.1f} GB")
    return "\n".join(lines)


@app.command()
def version():
    typer.echo("fusion-model-hub v1.0.0")


def cli_main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    app()


if __name__ == "__main__":
    cli_main()
