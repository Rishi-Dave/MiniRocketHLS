# Asks for Prith — network-saturation pivot (per April 23 meeting)

This file lists what we need from Prith and where each artifact should land
in the repo so the saturation harness + Phase 2 drop integration can
proceed. Drop everything into `saturation_harness/from_prith/` and update
the trailing checklist.

## 1. Packet-drop kernel/module
**Why:** without it, the network-overload behavior is "kernel hangs at
saturation" instead of the throughput plateau we want to plot. Per the
brief §4.3.

**What we need:**
- HLS or RTL source for a 512-bit AXI-Stream tail-drop (or RED) kernel that
  sits between `networklayer_1.M_AXIS_nl2sk` and the compute kernel's
  `input_timeseries` stream.
- AXI-Lite register map for at least: `drop_count` (running counter, u64),
  optional `queue_depth_high_water` (u32).
- Notes on:
  - drop policy (tail vs. RED; threshold)
  - clock domain / reset assumptions
  - any C-sim or co-sim test bench
  - which xclbins it has already been integrated into (and their accuracy)

**Land at:** `saturation_harness/from_prith/drop_kernel/`

## 2. Saturation experiment scripts (rate-generator)
**Why:** brief §11.3 — Prith already has data-generator scripts for variable
injection rates from prior saturation papers. We don't want to reinvent.

**What we need:** the scripts you used to drive prior throughput-vs-rate
plots — rate sweep, sample format, any pacing tricks (token bucket,
DPDK-equivalent, etc.).

**Land at:** `saturation_harness/from_prith/rate_scripts/`. We'll cherry-pick
ideas; our own driver is `saturation_harness/rate_sweep.py`.

## 3. JSON models for InsectSound and FruitFlies
**Why:** brief §7.2. The local copies in
`minirocket_modular/{InsectSound,FruitFlies}_minirocket_model.json` were
slow/error-prone to download. Need fresh copies for HYDRA + MultiRocket
axis variants too.

**Preferred path:** Wolverine path + commands to fetch (per the meeting,
"give you the path and you can just load him from there"). Wolverine paths
the harness can ingest directly:

| Variant         | Expected JSON path (local) |
|---|---|
| MiniRocket      | `minirocket_modular/<DATASET>_minirocket_model.json` (+ `_test_data.json`) |
| HYDRA           | `hydra_optimized/models/hydra_<dataset>_model.json` |
| MultiRocket     | `multirocket_optimized/models/multirocket84_<dataset>_model.json` |

Datasets needed: `InsectSound`, `MosquitoSound`, `FruitFlies` (all three
for all three algorithms = 9 model files + 9 test_data files).

**Land at:** the variant directories above. If too large for the repo, drop
in `saturation_harness/from_prith/models/` and we'll add symlinks.

## 4. (Optional) Reference saturation plots from prior papers
**Why:** to mirror axes / units / aesthetic in the lead paper figure
(brief §4.2 — the chart Prith showed during the meeting).

**Land at:** `saturation_harness/from_prith/reference_plots/`.

---

## Checklist (update inline once received)

- [ ] Drop kernel source + AXI-Lite register map
- [ ] Drop kernel test bench / co-sim
- [ ] Notes on drop policy (tail vs. RED) + threshold defaults
- [ ] Rate-generator scripts
- [ ] InsectSound model + test data (MiniRocket, HYDRA, MultiRocket)
- [ ] MosquitoSound model + test data (MiniRocket, HYDRA, MultiRocket)
- [ ] FruitFlies model + test data (MiniRocket, HYDRA, MultiRocket)
- [ ] Reference saturation plots (PDF/PNG) — optional

## Suggested Slack message (copy-paste)

> Hey Prith — kicking off the saturation pivot per the April 23 meeting.
> A few asks:
>
> 1. Could you share your packet-drop kernel + AXI-Lite register map? I'll
>    drop it into `MiniRocketHLS/saturation_harness/from_prith/drop_kernel/`.
> 2. Same for whatever rate-generator scripts you used for prior saturation
>    plots — I want to mirror the methodology. Even pseudocode is fine.
> 3. Models for InsectSound + FruitFlies (also MosquitoSound if quick) for
>    MiniRocket / HYDRA / MultiRocket — Wolverine path is ideal so I can
>    `wget`/`scp` directly. The local copies I have were giving me download
>    errors.
>
> No rush — I have the saturation harness scaffolded against the existing
> MiniRocket-axis xclbin and have csim-clean axis variants of HYDRA and
> MultiRocket queued for OCT this week. The drop module unblocks the
> *actual* plateau measurements — until then I'm running with synthetic
> drop accounting that overestimates losses.
