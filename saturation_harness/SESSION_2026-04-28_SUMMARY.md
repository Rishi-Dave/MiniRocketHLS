# 2026-04-28 — local prep session summary

Goal stated by user: "I have a build server and 2 FPGAs reserved for 2 days
on OCT. Do everything you need to do on this server before we run anything
on Wolverine." (Wolverine = current host. OCT = remote build cluster.)

What got finished locally so OCT time goes 100% to hw builds + experiments:

## Code shipped

### Saturation harness (Phase 1, ~1100 LOC, end-to-end smoke validated)
- [`saturation_harness/inference_adapters/base.py`](inference_adapters/base.py): InferenceAdapter ABC + `RunResult` dataclass (units convention: inferences/sec, µs latency, host-side or AXI-Lite drop counter)
- [`minirocket_axis_adapter.py`](inference_adapters/minirocket_axis_adapter.py): drives existing `fpga-network` xclbin over UDP, token-bucket pacing, dual-thread sender/receiver, FIFO matching for predictions
- [`hydra_axis_adapter.py`](inference_adapters/hydra_axis_adapter.py) + [`multirocket_axis_adapter.py`](inference_adapters/multirocket_axis_adapter.py): subclass MiniRocket-axis (same UDP wire format, distinct platform tags + model-loader factories)
- [`cpu_baseline_adapter.py`](inference_adapters/cpu_baseline_adapter.py) + [`cpu_servers/cpu_minirocket_server.py`](cpu_servers/cpu_minirocket_server.py): standalone UDP CPU MiniRocket server (one packet per sample, sample-id-keyed reply matching)
- [`rate_sweep.py`](rate_sweep.py): CLI with multi-schema dataset loader (MiniRocket / HYDRA / MultiRocket JSON layouts), log/linear rate schedules, JSONL output
- [`metrics.py`](metrics.py): per-run aggregator + saturation-point detection heuristic (`achieved/configured < 0.95` OR `drop_rate > 0.01`)
- [`plot_saturation.py`](plot_saturation.py): matplotlib throughput / drop-rate / latency curves
- [`runbook.md`](runbook.md): operational procedure + caveats

Smoke validation: 3-rate CPU sweep on InsectSound (1, 5, 25 inf/s) showed
target/achieved within 2%, 0% drops, p50≈2 ms, plots emitted. Two real bugs
caught & fixed (aeon API change, UDP request/reply socket-pair mismatch).

### HYDRA axis-stream HLS variant (Phase 3, OCT-buildable)
- [`hydra_optimized/hydra_axis/include/hydra_axis.hpp`](../hydra_optimized/hydra_axis/include/hydra_axis.hpp): `pkt = ap_axiu<512,1,1,16>`, 7 m_axi ports
- [`src/hydra_axis.cpp`](../hydra_optimized/hydra_axis/src/hydra_axis.cpp): static-helper port of HYDRA compute path with shift-register-on-stream ingress
- [`test/test_hydra_axis.cpp`](../hydra_optimized/hydra_axis/test/test_hydra_axis.cpp): smoke csim
- [`make.tcl`](../hydra_optimized/hydra_axis/make.tcl) + [`Makefile`](../hydra_optimized/hydra_axis/Makefile) + [`config_axis.cfg`](../hydra_optimized/hydra_axis/config_axis.cfg) + [`build_axis_xclbin.sh`](../hydra_optimized/hydra_axis/build_axis_xclbin.sh)
- **csim: PASS** (4/4 logits deterministic, 0 errors, 6 s)
- **csynth: PASS** at Fmax 379.75 MHz (above 300 MHz target)

### MultiRocket axis-stream HLS variant (Phase 5, OCT-buildable)
- [`multirocket_optimized/multirocket_axis/`](../multirocket_optimized/multirocket_axis/) — same structure as HYDRA axis (10 m_axi ports, kernel weights compiled-in from `weights.txt`, two-representation compute path)
- **csim: PASS** (4/4 logits deterministic, 0 errors, 30 s)
- **csynth: PASS** at Fmax 370.78 MHz (after fixing two HLS issues: removed `static` on locally-partitioned arrays, replaced address-of-partitioned-element pointer cast with `union`-based reinterpret in EMIT_PRED loop)

### MultiRocket m_axi rebuild prep (Phase 4)
- Confirmed source tree clean post Jan-6 algorithm fix (4 pooling ops, first-order diff, 5,376 features)
- Existing [`config.cfg`](../multirocket_optimized/config.cfg) supports rebuild as-is — 12 HBM ports
- xclbins on disk all stale (predate the fix), need fresh hw build

### Operational docs
- [`oct_parallel_builds.md`](oct_parallel_builds.md): exact `tmux new -d -s ...` commands + monitor-subagent invocations for the 3 parallel builds (MultiRocket m_axi rebuild, HYDRA-axis xclbin, MultiRocket-axis xclbin)
- [`from_prith/REQUEST_FROM_PRITH.md`](from_prith/REQUEST_FROM_PRITH.md): Slack-pasteable ask covering drop kernel + rate-generator scripts + Wolverine model paths

## Ready to run on OCT

| Build | Source state | Time est. | Output |
|---|---|---|---|
| MultiRocket m_axi rebuild | clean | 4–8 h | `multirocket_optimized/build_dir.hw.post_jan6/krnl.xclbin` |
| HYDRA-axis xclbin | csim+csynth PASS | ~30 min HLS + ~3–5 h v++ link | `hydra_optimized/hydra_axis/build_dir.hw/krnl.xclbin` |
| MultiRocket-axis xclbin | csim+csynth PASS | ~1 h HLS + ~5–8 h v++ link | `multirocket_optimized/multirocket_axis/build_dir.hw/krnl.xclbin` |

All three can run in parallel on a single OCT build node — the v++ link is
CPU+RAM bound, doesn't need the FPGA. Reserve the 2 FPGA nodes for
post-build saturation experiments (Phase 6), not builds.

## Still external-dependency-blocked

| Phase | Blocked by |
|---|---|
| 0.1 — Prith sync | Prith (request doc drafted) |
| 0.2 — OCT registration | user (procedure not on this server) |
| 0.3 — verify existing fpga-network xclbin against the harness | needs FPGA IP / port info; trivially runnable once you give me those |
| 0.4 — recover `minirocket_modular/host/src/` from git | this dir isn't a git repo on Wolverine; recover from the GitHub remote (`Rishi-Dave/MiniRocketHLS`) instead |
| 2 — drop kernel integration | Prith |
| 6 — saturation experiments on OCT | all builds finishing + Prith's drop module + OCT FPGA access |

## Headline metric for the paper plan

Resources / timing for each axis variant (300 MHz target, U280):

| Variant | Fmax | Build status |
|---|---|---|
| MiniRocket axis (existing fpga-network) | 397.6 MHz (per MEMORY.md) | hw xclbin built |
| **HYDRA axis (new)** | **379.75 MHz** | csynth PASS; ready for OCT v++ link |
| **MultiRocket axis (new)** | **370.78 MHz** | csynth PASS; ready for OCT v++ link |

All three clear the 300 MHz fence with comfortable margin. No timing-closure
work needed before kicking off the v++ link stage on OCT.
