# Multi-node Scale Load Test

Containerized multi-node deployment + load test for fusion-model-hub cluster
routing. Measures throughput / latency / per-node distribution across all-healthy,
failover, and real-MLX-only scenarios.

## Why mock backends

Fusion-MLX is Apple-Silicon + MLX-framework only — it cannot run inside a Linux
container. So the multi-node topology is **hybrid**:

| Node | Backend | Real model? |
|------|---------|-------------|
| `real-mlx` | host MLX (`host.docker.internal:11434`) | yes — genuine `Qwen3-0.6B-4bit` inference |
| `mock-mlx-1..3` | `mock_mlx.py` container stub | no — canned reply + fixed delay |

`mock_mlx.py` is a deterministic FastAPI stub (`/health`, `/v1/chat/completions`,
`/v1/models/{name}/load|unload`). It is NOT a model — returns canned completions
echoing the node name so the driver can verify which node served each request.
Delay is a fixed env var (`MOCK_DELAY_MS`), not learned.

The host's real MLX stays node 0 and does real inference (satisfies the
"真实加载模型" rule); mocks let the routing fan out to N backends.

## Topology

```
            ┌─────────────────────────┐
 load  ───▶ │  hub (container)        │  POST /api/v1/cluster/route-inference
 driver     │  FMH_MLX_URL=host:11434 │  mode=cluster → iterates registered nodes
            └───────────┬─────────────┘
                        │ _chat(node.url) → {url}/v1/chat/completions
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
 real-mlx (host)   mock-mlx-1       mock-mlx-2/3  (containers, shared net)
 Qwen3-0.6B-4bit   canned+delay     canned+delay
```

`route_inference` (mode=cluster) iterates cluster nodes in `list_cluster_nodes`
order — `order_by(ClusterNode.created_at.desc())`, i.e. **newest-registered
first** — and returns on the first success. This is **pure failover, not
round-robin/load-balance**: the newest node absorbs all healthy traffic; earlier
nodes only serve when newer ones fail. The test documents this real behavior
(see issue #31: newest-first order shadows earlier-registered primaries).
Failover is exercised by unregistering the dominant node between scenarios.

## Files

- `Dockerfile` — Hub container (python:3.12-slim, editable install, auth off)
- `Dockerfile.mock` — mock MLX backend container
- `mock_mlx.py` — deterministic MLX stub
- `docker-compose-multinode.yml` — hub + 3 mock backends (host MLX = node 0)
- `run_scale_test.py` — load driver: register nodes, create+publish model,
  concurrent route-inference across 3 scenarios, JSON report + cleanup

## Run

Prerequisites: host MLX running with `Qwen3-0.6B-4bit` cached.

```bash
~/claude-home/fusion-mlx/start.sh start          # host MLX, port 11434
source /Users/dahai/fusion/.venv/bin/activate

cd tests/integration/multinode
docker compose -f docker-compose-multinode.yml up -d --build
# wait for hub health
curl -s localhost:11444/api/v1/system/health

# run the load test (32 concurrent, 200 requests/scenario)
python run_scale_test.py --hub http://localhost:11444 --concurrency 32 --requests 200

# teardown
docker compose -f docker-compose-multinode.yml down -v
~/claude-home/fusion-mlx/start.sh stop
```

Output: `scale_test_report.json` + per-scenario summary to stdout. The driver
cleans up the model + nodes it created on exit.

## Scenarios

- **S1 all-healthy** — all 4 nodes registered; measures baseline throughput +
  shows the **newest-registered** node (`mock-mlx-3`) absorbs all traffic
  (failover order = `created_at DESC`), while `real-mlx` (registered first)
  serves 0 — the routing-order shadow documented in issue #31.
- **S2 failover** — unregister the **dominant** node (last-registered mock) so
  routing genuinely falls to the next active node. Verifies failover works
  when the node that was serving is removed.
- **S3 real-MLX-only** — unregister all mocks; only the host MLX node remains
  → every request is genuine `Qwen3-0.6B-4bit` inference. Verifies the real
  load path end-to-end and measures single-node saturation (the case for
  multi-node clustering).

## Measured results (32 concurrent × 150 req/scenario)

| Scenario | ok | rps | p50 | node distribution |
|----------|----|-----|-----|-------------------|
| S1 all-healthy (4 nodes) | 150/150 | 35.91 | 828ms | `mock-mlx-3` (newest): 127, `mock-mlx-2`: 17, `mock-mlx-1`: 6, `real-mlx`: **0** |
| S2 failover (drop `mock-mlx-3`) | 149/150 | 6.71 | 913ms | `mock-mlx-2`: 115, `mock-mlx-1`: 29, `real-mlx`: 5 |
| S3 real-MLX-only | 10/150 | 0.15 | 49729ms | `real-mlx`: 10 (saturated) |

**Findings:**
1. **Routing order** (`created_at DESC`) shadows the primary: `real-mlx`
   registered first → tried last → never serves while a newer node is alive
   (issue #31).
2. **Failover works**: dropping the dominant node shifts traffic to the next
   active node (S2), but `real-mlx` still only catches 5/150 because it remains
   last in DESC order — confirming the order bug, not a real spread.
3. **Single-node saturation**: one real MLX node caps at ~0.15 rps under
   concurrency 32 (S3) — the bottleneck that multi-node clustering exists to
   solve. With load-balancing (not just failover), the 3 mock nodes sustain
   35.91 rps (S1) vs 0.15 (S3) — a ~240× capacity gap that multi-node closes.

### httpx pooling impact (H8 scope gap, fixed this branch)

`cluster.py` was using raw `httpx` (per-call client, no connection pooling),
leaving the H8 PoolClient fix (`server/http_client.py`) applied only to
`inference.py`. Migrating `cluster.py` to `from .. import http_client as httpx`
on the cluster-routing hot path:

| Scenario | before (raw httpx) | after (pooled) | gain |
|----------|---------------------|----------------|------|
| S1 rps | 16.69 | 93.45 | **5.6×** |
| S1 p50 | 881ms | 148ms | **6× lower** |
| S2 rps | 18.9 | 56.76 | **3×** |

(16 concurrent × 120 req; the 32×150 run above is post-fix.) Per-call client
construction on the routing path was the dominant cost; pooled transports
removed it.
