#!/usr/bin/env python3
"""Multi-node scale load test driver.

Registers N MLX backend nodes (1 real host MLX + M mock containers) as cluster
nodes in the containerized Hub, creates + publishes a model, then fires
concurrent /cluster/route-inference requests and measures throughput / latency
/ per-node distribution across three scenarios:

  S1 all-healthy   — every node alive; route_inference returns on node 0
  S2 failover      — kill node 0 (mock), routing falls to next node
  S3 real-MLX-only — kill all mocks, only the host's real MLX node serves
                     (genuine model inference, per the "真实加载模型" rule)

Usage:
    python run_scale_test.py --hub http://localhost:11444 --concurrency 32 --requests 200

Outputs a JSON report + human summary to stdout and a log file. Cleans up
created model/nodes on exit (process data rule: keep only final outputs+logs).
"""
import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter

import httpx

# --- config placeholders, filled below ---
HUB_URL = "http://localhost:11444"
CONCURRENCY = 32
TOTAL_REQUESTS = 200
REAL_MLX_URL = "http://host.docker.internal:11434"
REAL_MODEL_ID = "Qwen3-0.6B-4bit"  # cached MLX model id, verified loadable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [scale-test] %(message)s",
)
logger = logging.getLogger("scale-test")


class HubClient:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    async def register_node(self, client, name, url, caps="inference"):
        r = await client.post(
            f"{self.base}/api/v1/cluster/nodes",
            json={"name": name, "url": url, "capabilities": caps},
            timeout=15,
        )
        r.raise_for_status()
        node = r.json()
        logger.info("registered node: id=%s name=%s url=%s", node["id"], name, url)
        return node

    async def heartbeat(self, client, node_id):
        r = await client.post(
            f"{self.base}/api/v1/cluster/nodes/{node_id}/heartbeat", timeout=10
        )
        r.raise_for_status()

    async def delete_node(self, client, node_id):
        await client.delete(f"{self.base}/api/v1/cluster/nodes/{node_id}", timeout=10)

    async def create_model(self, client, name, hf_repo):
        r = await client.post(
            f"{self.base}/api/v1/models",
            json={"name": name, "hf_repo": hf_repo, "description": "scale-test"},
            timeout=15,
        )
        if r.status_code == 409:
            logger.info("model exists, fetching: %s", name)
            r2 = await client.get(f"{self.base}/api/v1/models", params={"keyword": name}, timeout=10)
            for m in r2.json().get("items", r2.json().get("data", [])):
                if m["name"] == name:
                    return m
        r.raise_for_status()
        m = r.json()
        logger.info("created model: id=%s name=%s", m["id"], name)
        return m

    async def publish_model(self, client, model_id):
        r = await client.post(f"{self.base}/api/v1/models/{model_id}/publish", timeout=15)
        if r.status_code == 409:
            logger.info("model already published: %s", model_id)
            return
        r.raise_for_status()
        logger.info("published model: %s", model_id)

    async def delete_model(self, client, model_id):
        await client.delete(f"{self.base}/api/v1/models/{model_id}", timeout=10)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hub", default=os.environ.get("HUB_URL", "http://localhost:11444"))
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--real-mlx-url", default=os.environ.get("REAL_MLX_URL", REAL_MLX_URL))
    p.add_argument("--model-id", default=os.environ.get("REAL_MODEL_ID", REAL_MODEL_ID))
    p.add_argument("--mock-nodes", default="mock-mlx-1:11434,mock-mlx-2:11434,mock-mlx-3:11434")
    p.add_argument("--out", default="scale_test_report.json")
    return p.parse_args()


async def one_request(client, hub_url, model_id, req_idx):
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{hub_url}/api/v1/cluster/route-inference",
            json={
                "model_id": model_id,
                "messages": [{"role": "user", "content": f"load req {req_idx}: reply pong"}],
                "mode": "cluster",
            },
            timeout=150,
        )
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "ms": dt, "routedTo": None,
                    "error": r.text[:120]}
        body = r.json()
        return {"ok": True, "status": 200, "ms": dt,
                "routedTo": body.get("routedTo"), "content": body.get("content", "")[:40]}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        return {"ok": False, "status": 0, "ms": dt, "routedTo": None, "error": str(e)[:120]}


async def run_load(hub_url, model_id, concurrency, total):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(idx):
        async with sem:
            async with httpx.AsyncClient() as c:
                return await one_request(c, hub_url, model_id, idx)

    t_start = time.perf_counter()
    tasks = [asyncio.create_task(bounded(i)) for i in range(total)]
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    wall = time.perf_counter() - t_start
    return results, wall


def summarize(results, wall):
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    lat = [r["ms"] for r in ok]
    dist = Counter(r["routedTo"] for r in ok)
    return {
        "total": len(results),
        "ok": len(ok),
        "failed": len(fail),
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(len(ok) / wall, 2) if wall > 0 else 0,
        "latency_ms": {
            "p50": round(statistics.median(lat), 2) if lat else 0,
            "p95": round(_percentile(lat, 95), 2) if lat else 0,
            "p99": round(_percentile(lat, 99), 2) if lat else 0,
            "mean": round(statistics.mean(lat), 2) if lat else 0,
            "min": round(min(lat), 2) if lat else 0,
            "max": round(max(lat), 2) if lat else 0,
        },
        "node_distribution": dict(dist),
        "errors": fail[:5],
    }


def _percentile(data, pct):
    if not data:
        return 0
    s = sorted(data)
    k = max(0, min(len(s) - 1, round((pct / 100.0) * (len(s) - 1))))
    return s[k]


async def setup(hub, client, args):
    # Register real MLX as a cluster node (route_inference mode=cluster only
    # uses registered cluster nodes, NOT the local MLX). Then mock nodes.
    nodes = []
    real = await hub.register_node(client, "real-mlx", args.real_mlx_url, caps="inference")
    nodes.append(("real-mlx", real["id"]))
    for spec in [s for s in args.mock_nodes.split(",") if s.strip()]:
        host, _, port = spec.partition(":")
        # Hub container resolves mock service names on the shared network.
        url = f"http://{host}:{port or 11434}"
        n = await hub.register_node(client, host, url, caps="inference")
        nodes.append((host, n["id"]))
    # Heartbeat all nodes so they are "active" (not reaped as stale).
    for _, nid in nodes:
        await hub.heartbeat(client, nid)
    # Create + publish the model. hf_repo == real MLX model id so _chat sends
    # the right model_name to the real node; mocks ignore it.
    model = await hub.create_model(client, f"scale-test-{int(time.time())%100000}", args.model_id)
    await hub.publish_model(client, model["id"])
    return model["id"], nodes


async def cleanup(hub, client, model_id, nodes):
    for _name, nid in nodes:
        try:
            await hub.delete_node(client, nid)
        except Exception:
            logger.warning("cleanup: failed to delete node %s", nid, exc_info=True)
    try:
        await hub.delete_model(client, model_id)
    except Exception:
        logger.warning("cleanup: failed to delete model %s", model_id, exc_info=True)


async def main_async(args):
    hub = HubClient(args.hub)
    report = {"args": vars(args), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenarios": {}}
    async with httpx.AsyncClient() as client:
        # wait for hub health
        for _ in range(30):
            try:
                r = await client.get(f"{args.hub}/api/v1/system/health", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            logger.error("Hub not healthy at %s", args.hub)
            return 1

        model_id, nodes = await setup(hub, client, args)
        report["model_id"] = model_id
        report["nodes"] = [{"name": n, "id": i} for n, i in nodes]
        logger.info("setup done: model=%s nodes=%d", model_id, len(nodes))

        try:
            # S1: all healthy
            logger.info("=== S1 all-healthy ===")
            res, wall = await run_load(args.hub, model_id, args.concurrency, args.requests)
            report["scenarios"]["S1_all_healthy"] = summarize(res, wall)
            _print_scenario("S1 all-healthy", report["scenarios"]["S1_all_healthy"])

            # S2: failover — unregister the DOMINANT node (last-registered mock,
            # which route_inference tries first under created_at DESC order) so
            # routing genuinely falls to the next active node. Unregistering a
            # node that served 0 reqs (e.g. real-mlx, first-registered) is a
            # no-op test. Pick nodes[-1] = last registered mock = dominant.
            fail_node = nodes[-1]
            logger.info("=== S2 failover (unregister dominant %s) ===", fail_node[0])
            await hub.delete_node(client, fail_node[1])
            nodes.pop()
            res, wall = await run_load(args.hub, model_id, args.concurrency, args.requests)
            report["scenarios"]["S2_failover"] = summarize(res, wall)
            _print_scenario("S2 failover", report["scenarios"]["S2_failover"])

            # S3: real-MLX-only — unregister all mocks, only real MLX remains.
            logger.info("=== S3 real-MLX-only ===")
            killed = []
            for name, nid in nodes[1:]:
                await hub.delete_node(client, nid)
                killed.append((name, nid))
            res, wall = await run_load(args.hub, model_id, args.concurrency, args.requests)
            report["scenarios"]["S3_real_mlx_only"] = summarize(res, wall)
            _print_scenario("S3 real-MLX-only", report["scenarios"]["S3_real_mlx_only"])
            # verify at least one real-MLX reply present
            real_hits = report["scenarios"]["S3_real_mlx_only"]["node_distribution"]
            logger.info("S3 node distribution (expect real-mlx): %s", real_hits)
        finally:
            await cleanup(hub, client, model_id, nodes)

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("report written: %s", args.out)
    return 0


def _print_scenario(name, s):
    print(f"\n--- {name} ---")
    print(f"  ok={s['ok']}/{s['total']}  failed={s['failed']}  wall={s['wall_seconds']}s  rps={s['throughput_rps']}")
    print(f"  latency ms: p50={s['latency_ms']['p50']} p95={s['latency_ms']['p95']} p99={s['latency_ms']['p99']} mean={s['latency_ms']['mean']}")
    print(f"  node distribution: {s['node_distribution']}")
    if s["errors"]:
        print(f"  sample errors: {s['errors'][:2]}")


def main():
    args = parse_args()
    global HUB_URL, CONCURRENCY, TOTAL_REQUESTS, REAL_MLX_URL, REAL_MODEL_ID
    HUB_URL, CONCURRENCY, TOTAL_REQUESTS = args.hub, args.concurrency, args.requests
    REAL_MLX_URL, REAL_MODEL_ID = args.real_mlx_url, args.model_id
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
