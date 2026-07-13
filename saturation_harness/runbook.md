# Saturation experiment runbook

This is the operational runbook for the network-saturation experiments
introduced by the April 23, 2026 paper-revisions meeting. The goal is to
produce throughput-vs-injection-rate curves comparing CPU and FPGA
(MiniRocket / HYDRA / MultiRocket) inference under the same UDP-based
network harness. See [`/home/rdave009/.claude/plans/i-need-to-start-curried-firefly.md`](../../.claude/plans/i-need-to-start-curried-firefly.md)
for the full plan.

## Layout

```
saturation_harness/
  inference_adapters/
    base.py                       # InferenceAdapter ABC + RunResult
    minirocket_axis_adapter.py    # FPGA UDP, beat-per-float wire format
    cpu_baseline_adapter.py       # CPU UDP, packet-per-sample wire format
    hydra_axis_adapter.py         # (Phase 3)
    multirocket_axis_adapter.py   # (Phase 5)
  cpu_servers/
    cpu_minirocket_server.py      # standalone UDP MiniRocket inference server
  rate_sweep.py                   # CLI driver: sweep injection rates, write JSONL
  metrics.py                      # aggregate + saturation-point detection
  plot_saturation.py              # throughput / drop / latency plots
  runs/                           # gitignored: per-experiment output
  from_prith/                     # gitignored: drop-module / scripts / models
```

## Wire formats at a glance

| Adapter | Per-sample wire | Reply | Sample matching |
|---|---|---|---|
| `minirocket-axis` | L UDP packets, each 64 B (one float in low 4 B) | L × C beats; last C carry the prediction | FIFO order |
| `cpu-minirocket`  | 1 UDP packet (`<u32 sample_id><float×L>`) | 1 UDP packet (`<u32 sample_id><float×C>`) | `sample_id` |

These are intentionally different: the FPGA pipeline streams one beat per
sample point because that matches its sliding-window kernel; the CPU server
takes a whole sample per packet because that's how a real CPU inference
server would be deployed. Both are valid “networked inference” designs; what
the paper compares is **the rate at which each platform stops producing
predictions matching the injection rate**.

## Smoke run (Phase 1, no FPGA drop module yet)

CPU baseline only — runs entirely on localhost, no FPGA needed.

```bash
cd MiniRocketHLS/saturation_harness
python3 rate_sweep.py \
    --adapter cpu-minirocket \
    --dataset GunPoint \
    --model-dir ../minirocket_modular \
    --rate-min 1 --rate-max 1000 --rate-steps 10 --rate-mode log \
    --duration-s 5 --warmup-s 1 \
    --out runs/smoke_phase1/cpu_GunPoint
```

FPGA-MiniRocket — requires the fpga-network xclbin running on a U280 reachable
over UDP at `<FPGA_IP>:<FPGA_PORT>`, and replies coming back to the host's
`<LISTEN_PORT>`. **Phase 1 caveat: there is no on-FPGA drop module yet, so
overload may stall the kernel rather than plateau** (see Phase 2).

```bash
python3 rate_sweep.py \
    --adapter minirocket-axis \
    --dataset GunPoint \
    --model-dir ../minirocket_modular \
    --fpga-ip 192.168.40.2 --fpga-port 62177 --listen-port 62178 \
    --rate-min 100 --rate-max 50000 --rate-steps 12 --rate-mode log \
    --duration-s 5 --warmup-s 1 \
    --out runs/smoke_phase1/fpga_minirocket_GunPoint
```

Aggregate and plot:

```bash
python3 metrics.py \
    --runs runs/smoke_phase1/* \
    --out runs/smoke_phase1/aggregate

python3 plot_saturation.py \
    --aggregate runs/smoke_phase1/aggregate/aggregate.jsonl \
    --out runs/smoke_phase1/plots
```

Expected shape (per the meeting brief §4.2):
- CPU adapter: throughput plateau between 10–100 inferences/sec.
- FPGA-MiniRocket adapter: plateau ~800–3000 inferences/sec depending on
  dataset (longer series → lower throughput).
- Drop-rate plot: zero/near-zero below saturation, climbs through saturation,
  saturates at 1.0 if pushed far past the platform's ceiling.

## Common pitfalls

- **The FPGA adapter assumes FIFO matching of replies.** If overload causes
  some beats to be dropped *mid-sample*, the matching will go off by one and
  every subsequent latency reading will be wrong (throughput is still
  correct because we count completed predictions, not matched ones).
  Phase 2 fixes this by reading the FPGA AXI-Lite drop register and
  resyncing the FIFO. Until then, treat per-sample latency as a
  below-saturation measurement only.
- **CPU server uses aeon's MiniRocket transformer fitted on a synthetic
  dummy.** Throughput is realistic; accuracy reported by the CPU adapter is
  not. Don't compare CPU `accuracy` numbers to FPGA `accuracy` numbers in the
  paper — that's not what the CPU baseline measures. Cross-check accuracy
  with `cpu/cpu-latency.py` separately.
- **UDP receive-buffer drops at the host.** Increase
  `net.core.rmem_max` / `net.core.rmem_default` (`sudo sysctl -w
  net.core.rmem_max=134217728`) and pass `--cpu-listen-port` / `--listen-port`
  to a port not used by another process.
- **Python pacing ceiling.** The token-bucket loop in user-space tops out
  somewhere between 100k and 1M packets/sec depending on host. If you need
  rates above that (unlikely for the saturation paper since FPGA itself
  saturates well below), escalate to a Cython hot loop or DPDK.

## Memory updates

After each smoke run, update the variant-status table in
`~/.claude/projects/-home-rdave009-minirocket-hls/memory/MEMORY.md` with the
saturation point per platform (rate at which `achieved/configured < 0.95`).

## Phase progression

- **Phase 1 (this directory):** harness scaffolding + MiniRocket-axis + CPU
  adapters. Smoke-runnable today; FPGA drop module not yet wired.
- **Phase 2:** integrate Prith's drop kernel into `fpga-network/config.cfg`;
  extend the MiniRocket-axis adapter to read drop_count via XRT AXI-Lite.
- **Phase 3:** add `hydra_axis_adapter.py` and the `hydra_optimized/hydra_axis/`
  HLS variant.
- **Phase 5:** add `multirocket_axis_adapter.py` and the
  `multirocket_optimized/multirocket_axis/` HLS variant.
- **Phase 6:** OCT 2-node experiments, paper figures.
