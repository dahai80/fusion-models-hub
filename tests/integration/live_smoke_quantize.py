"""Live fusion-mlx smoke test: verify converter.quantize async-job contract.

Exercises the rewritten converter against a REAL fusion-mlx on 11434:
  POST /v1/quantize {model, output_path, quant_bits} -> {job_id, queued}
  poll GET /v1/quantize/jobs/{id} until status=="done"
  result carries output_path; provenance (hash/size) computed locally.

Cleans the produced output dir. Run with real MLX up.

Usage: python tests/integration/live_smoke_quantize.py
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("live_smoke")

MLX_MODELS = os.path.expanduser("~/.fusion-mlx/models")
SOURCE_MODEL = os.path.join(MLX_MODELS, "mlx-community-Llama-3.2-1B-Instruct-4bit")


def _mlx_key() -> str:
    import json

    with open(os.path.expanduser("~/.fusion-mlx/settings.json")) as f:
        return json.load(f).get("auth", {}).get("api_key", "")


def cleanup(out_path: str):
    import shutil

    if out_path and os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
        log.info("[cleanup] removed %s", out_path)


async def run() -> bool:
    from fusion_model_hub.convert.converter import ModelConverter

    # Pre-clean any stale MLX outputs (MLX rejects an existing output dir).
    cleanup(os.path.join(MLX_MODELS, "mlx-community-Llama-3-8bit.mlx"))
    cleanup(os.path.join(MLX_MODELS, "llama-3.2-1b-live-smoke-8bit.mlx"))
    if not os.path.isdir(SOURCE_MODEL):
        log.error("[smoke] source model dir not found: %s", SOURCE_MODEL)
        return False
    conv = ModelConverter(mlx_url="http://127.0.0.1:11434", api_key=_mlx_key())
    log.info("[smoke] quantize source=%s bits=8", SOURCE_MODEL)
    result = await conv.quantize(SOURCE_MODEL, bits=8)
    log.info("[smoke] converter result: %s", result)
    ok = result.get("status") == "completed" and bool(result.get("output_path"))
    if ok and result.get("file_hash"):
        log.info("[smoke] provenance hash=%s size=%s", result["file_hash"][:16], result.get("file_size"))
    cleanup(result.get("output_path", ""))
    return ok


def main() -> int:
    import asyncio

    ok = asyncio.run(run())
    log.info("RESULT quantize_live=%s", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
