# Deterministic mock MLX backend for the multi-node scale test.
# Real MLX cannot run in a Linux container (MLX framework is Apple-Silicon
# only), so the extra cluster nodes are this stub. It is NOT a model — it
# returns canned chat completions with a configurable fixed delay (Rule 5:
# decide with code, not tokens; the delay is deterministic, not learned). The
# host's real MLX remains node 0 and does genuine model inference.
import asyncio
import logging
import os
import time

from fastapi import FastAPI, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [mock-mlx:%(name)s] %(message)s",
)
logger = logging.getLogger("mock-mlx")

app = FastAPI(title="mock-mlx")

# Simulated per-request inference latency (ms). Smaller = more throughput.
DELAY_MS = int(os.environ.get("MOCK_DELAY_MS", "20"))
NODE_NAME = os.environ.get("MOCK_NODE_NAME", "mock-mlx")


@app.get("/health")
async def health():
    return {"status": "healthy", "ready": True, "model_loaded": True, "node": NODE_NAME}


@app.get("/v1/models")
async def list_models():
    # Expose one fake model so the Hub's model-name match resolves; the Hub
    # looks up model_name from its own DB (hf_repo or name), so this list is
    # informational — route_inference just POSTs /v1/chat/completions.
    return {"data": [{"id": NODE_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "unknown")
    messages = body.get("messages", [])
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break
    # Deterministic canned reply — echoes the node + first chars of input so
    # the load driver can verify WHICH node served each request (distribution).
    echo = (user_msg[:32] + "...") if len(user_msg) > 32 else user_msg
    if DELAY_MS > 0:
        await asyncio.sleep(DELAY_MS / 1000.0)
    # Token-ish counts derived deterministically from input length (no model).
    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in messages)
    completion_tokens = 8
    return {
        "id": f"chatcmpl-mock-{NODE_NAME}-{int(time.time() * 1000) % 100000}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"[{NODE_NAME}] mock-pong: {echo}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/models/{name}/load")
async def load_model(name: str):
    logger.info("load (no-op): model=%s", name)
    return {"loaded": True, "model": name, "node": NODE_NAME}


@app.post("/v1/models/{name}/unload")
async def unload_model(name: str):
    logger.info("unload (no-op): model=%s", name)
    return {"unloaded": True, "model": name, "node": NODE_NAME}


@app.get("/v1/metrics/json")
async def metrics_json():
    return {"node": NODE_NAME, "mock": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MOCK_PORT", "11435"))
    logger.info("mock-mlx starting: node=%s port=%d delay_ms=%d", NODE_NAME, port, DELAY_MS)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
