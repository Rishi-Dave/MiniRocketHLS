#!/usr/bin/env python3
"""Saturation rate-sweep driver.

For a given (adapter, dataset), sweep injection rates from ``--rate-min`` to
``--rate-max`` over ``--rate-steps`` points (linear or log spacing) and
record one ``RunResult`` per rate as a JSONL row in the output directory.

Output layout::

    <out>/
        sweep.jsonl       # one RunResult per rate
        sweep_meta.json   # CLI args, timestamps, host info, model info
        cli.txt           # exact command line, for replay

Example::

    python3 rate_sweep.py \\
        --adapter minirocket-axis \\
        --dataset GunPoint \\
        --model-dir ../minirocket_modular \\
        --fpga-ip 192.168.40.2 --fpga-port 62177 --listen-port 62178 \\
        --rate-min 100 --rate-max 50000 --rate-steps 12 --rate-mode log \\
        --duration-s 5 \\
        --out runs/smoke_phase1/minirocket_axis_GunPoint

    python3 rate_sweep.py \\
        --adapter cpu-minirocket \\
        --dataset GunPoint \\
        --model-dir ../minirocket_modular \\
        --rate-min 1 --rate-max 1000 --rate-steps 12 --rate-mode log \\
        --duration-s 5 \\
        --out runs/smoke_phase1/cpu_minirocket_GunPoint
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import platform
import socket
import sys
import time
from typing import List, Optional  # noqa: F401

# Local imports (works when invoked as module or as script).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from inference_adapters.base import AdapterConfig, InferenceAdapter, RunResult  # noqa: E402


def _load_dataset(model_dir: str, name: str, override_path: Optional[str] = None):
    """Load test data for a dataset. Supports the three known schemas:

    - MiniRocket (minirocket_modular): ``<DATASET>_minirocket_model_test_data.json``
      with ``X_test`` / ``y_test`` / ``series_length``.
    - HYDRA (hydra_optimized/models): ``hydra_<dataset>_test{,_1000,_*}.json``
      with ``time_series`` / ``labels`` / ``time_series_length``.
    - MultiRocket (multirocket_optimized/models): ``multirocket84_<dataset>_test.json``
      same shape as HYDRA.

    If ``override_path`` is set, load that file directly and auto-detect schema.
    """
    if override_path:
        p = pathlib.Path(override_path)
    else:
        candidates = [
            pathlib.Path(model_dir) / f"{name}_minirocket_model_test_data.json",
            pathlib.Path(model_dir) / f"hydra_{name.lower()}_test_1000.json",
            pathlib.Path(model_dir) / f"hydra_{name.lower()}_test.json",
            pathlib.Path(model_dir) / f"multirocket84_{name.lower()}_test.json",
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            raise FileNotFoundError(
                f"no test-data JSON found for dataset='{name}' under model_dir='{model_dir}'. "
                f"Tried: {[str(c) for c in candidates]}. "
                f"Pass --test-data-json explicitly to override."
            )
    with open(p) as f:
        d = json.load(f)
    import numpy as np
    if "X_test" in d:  # MiniRocket schema
        X = np.asarray(d["X_test"], dtype=np.float32)
        y = np.asarray(d["y_test"], dtype=np.int32)
        L = int(d.get("series_length", d.get("time_series_length")))
    elif "time_series" in d:  # HYDRA / MultiRocket schema
        X = np.asarray(d["time_series"], dtype=np.float32)
        raw_labels = d["labels"]
        # HYDRA labels are sometimes strings (class names like 'aedes_female').
        # Map strings to integer indices by sorted unique order. Documented
        # ambiguity: this assumes the FPGA classifier's argmax-class output
        # uses the same ordering, which holds for models trained the same way
        # (the model JSON's `classes` field carries the canonical order).
        try:
            y = np.asarray(raw_labels, dtype=np.int32)
        except (TypeError, ValueError):
            unique_labels = sorted(set(raw_labels))
            label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
            y = np.asarray([label_to_idx[v] for v in raw_labels], dtype=np.int32)
        L = int(d["time_series_length"])
    else:
        raise ValueError(f"unrecognized test-data JSON schema in {p}: keys={list(d.keys())}")
    C = int(d["num_classes"])
    return X, y, L, C


def _build_rate_schedule(rmin: float, rmax: float, n: int, mode: str) -> List[float]:
    if n < 1:
        raise ValueError("rate-steps must be >= 1")
    if n == 1:
        return [rmin]
    if mode == "log":
        if rmin <= 0:
            raise ValueError("rate-min must be > 0 for log mode")
        lo, hi = math.log(rmin), math.log(rmax)
        return [math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    elif mode == "linear":
        return [rmin + (rmax - rmin) * i / (n - 1) for i in range(n)]
    else:
        raise ValueError(f"unknown rate-mode: {mode}")


def _build_adapter(args, series_length: int, num_classes: int) -> InferenceAdapter:
    if args.adapter == "minirocket-axis":
        from inference_adapters.minirocket_axis_adapter import MinirocketAxisAdapter
        if not args.fpga_ip or not args.fpga_port or not args.listen_port:
            raise SystemExit("--fpga-ip, --fpga-port, --listen-port required for minirocket-axis")
        return MinirocketAxisAdapter(
            fpga_ip=args.fpga_ip,
            fpga_port=args.fpga_port,
            listen_port=args.listen_port,
            series_length=series_length,
            num_classes=num_classes,
            dataset=args.dataset,
        )
    elif args.adapter == "hydra-axis":
        from inference_adapters.hydra_axis_adapter import HydraAxisAdapter
        if not args.fpga_ip or not args.fpga_port or not args.listen_port:
            raise SystemExit("--fpga-ip, --fpga-port, --listen-port required for hydra-axis")
        return HydraAxisAdapter(
            fpga_ip=args.fpga_ip,
            fpga_port=args.fpga_port,
            listen_port=args.listen_port,
            series_length=series_length,
            num_classes=num_classes,
            dataset=args.dataset,
        )
    elif args.adapter == "multirocket-axis":
        from inference_adapters.multirocket_axis_adapter import MultirocketAxisAdapter
        if not args.fpga_ip or not args.fpga_port or not args.listen_port:
            raise SystemExit("--fpga-ip, --fpga-port, --listen-port required for multirocket-axis")
        return MultirocketAxisAdapter(
            fpga_ip=args.fpga_ip,
            fpga_port=args.fpga_port,
            listen_port=args.listen_port,
            series_length=series_length,
            num_classes=num_classes,
            dataset=args.dataset,
        )
    elif args.adapter == "cpu-minirocket":
        from inference_adapters.cpu_baseline_adapter import CpuMinirocketAdapter
        model_json = pathlib.Path(args.model_dir) / f"{args.dataset}_minirocket_model.json"
        return CpuMinirocketAdapter(
            model_json=str(model_json),
            dataset=args.dataset,
            series_length=series_length,
            num_classes=num_classes,
            server_host=args.cpu_server_host,
            server_port=args.cpu_server_port,
            listen_port=args.cpu_listen_port,
            spawn_server=not args.cpu_no_spawn,
        )
    else:
        raise SystemExit(f"unknown adapter: {args.adapter}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    choices=["minirocket-axis", "cpu-minirocket",
                             "hydra-axis", "multirocket-axis"],
                    help="which adapter to drive (hydra-axis, multirocket-axis added in Phases 3/5)")
    ap.add_argument("--dataset", required=True,
                    help="dataset name; the harness expects "
                         "<model-dir>/<dataset>_minirocket_model{,_test_data}.json")
    ap.add_argument("--model-dir", default="../minirocket_modular",
                    help="directory containing test-data JSON; auto-detects schema across "
                         "MiniRocket / HYDRA / MultiRocket layouts")
    ap.add_argument("--test-data-json", default=None,
                    help="explicit path to test-data JSON, bypasses auto-detection")
    # Rate schedule
    ap.add_argument("--rate-min", type=float, required=True, help="lowest injection rate (inf/sec)")
    ap.add_argument("--rate-max", type=float, required=True, help="highest injection rate (inf/sec)")
    ap.add_argument("--rate-steps", type=int, default=12)
    ap.add_argument("--rate-mode", choices=["log", "linear"], default="log")
    # Per-rate timing
    ap.add_argument("--duration-s", type=float, default=5.0)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--cooldown-s", type=float, default=0.5)
    # FPGA-axis adapter wiring
    ap.add_argument("--fpga-ip", default=None)
    ap.add_argument("--fpga-port", type=int, default=None)
    ap.add_argument("--listen-port", type=int, default=None)
    # CPU adapter wiring
    ap.add_argument("--cpu-server-host", default="127.0.0.1")
    ap.add_argument("--cpu-server-port", type=int, default=5050)
    ap.add_argument("--cpu-listen-port", type=int, default=5051)
    ap.add_argument("--cpu-no-spawn", action="store_true",
                    help="don't spawn cpu_minirocket_server.py; assume already running")
    # Output / sample limits
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of distinct dataset samples (0 = all)")
    ap.add_argument("--out", required=True, help="output directory (will be created)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, series_length, num_classes = _load_dataset(args.model_dir, args.dataset, args.test_data_json)
    if args.limit and args.limit < X.shape[0]:
        X = X[:args.limit]
        y = y[:args.limit]
    samples = [list(map(float, row)) for row in X]
    labels = list(map(int, y))

    rates = _build_rate_schedule(args.rate_min, args.rate_max, args.rate_steps, args.rate_mode)

    # Persist meta + cli for reproducibility.
    meta = {
        "dataset": args.dataset,
        "adapter": args.adapter,
        "model_dir": str(pathlib.Path(args.model_dir).resolve()),
        "series_length": series_length,
        "num_classes": num_classes,
        "n_samples_dataset": len(samples),
        "rate_min": args.rate_min,
        "rate_max": args.rate_max,
        "rate_steps": args.rate_steps,
        "rate_mode": args.rate_mode,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "cooldown_s": args.cooldown_s,
        "rates_pps": rates,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        },
        "ts_start_unix": time.time(),
    }
    with open(out_dir / "sweep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "cli.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    sweep_path = out_dir / "sweep.jsonl"
    print(f"[sweep] adapter={args.adapter} dataset={args.dataset} "
          f"L={series_length} C={num_classes} rates={len(rates)} "
          f"({rates[0]:.1f}..{rates[-1]:.1f} pps, mode={args.rate_mode})  "
          f"per-rate={args.duration_s}s",
          flush=True)

    adapter = _build_adapter(args, series_length, num_classes)
    rc = 0
    try:
        with adapter, open(sweep_path, "w") as fout:
            for i, rate in enumerate(rates, 1):
                cfg = AdapterConfig(
                    rate_pps=rate,
                    duration_s=args.duration_s,
                    warmup_s=args.warmup_s,
                    cooldown_s=args.cooldown_s,
                )
                t0 = time.time()
                try:
                    res: RunResult = adapter.run(samples, labels, cfg)
                except Exception as e:  # noqa: BLE001 — log and continue sweep
                    rc = 1
                    print(f"[sweep] {i}/{len(rates)} rate={rate:.1f} ERROR: {e}", flush=True)
                    fout.write(json.dumps({
                        "platform": adapter.platform,
                        "dataset": args.dataset,
                        "configured_rate_pps": rate,
                        "error": repr(e),
                        "ts_unix": time.time(),
                    }) + "\n")
                    fout.flush()
                    continue
                fout.write(res.to_jsonl() + "\n")
                fout.flush()
                print(
                    f"[sweep] {i:>2}/{len(rates)} "
                    f"target={rate:>9.1f}  "
                    f"achieved={res.achieved_throughput_inf_per_s:>9.1f}  "
                    f"drop={res.drop_rate*100:>5.1f}%  "
                    f"p50={res.latency_us_p50:>7.1f}µs  "
                    f"p99={res.latency_us_p99:>7.1f}µs  "
                    f"acc={(res.accuracy*100 if not (res.accuracy != res.accuracy) else float('nan')):>5.1f}%  "
                    f"({time.time()-t0:.1f}s)",
                    flush=True,
                )
    finally:
        with open(out_dir / "sweep_meta.json", "r+") as f:
            meta = json.load(f)
            meta["ts_end_unix"] = time.time()
            f.seek(0)
            json.dump(meta, f, indent=2)
            f.truncate()
    print(f"[sweep] done -> {sweep_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
