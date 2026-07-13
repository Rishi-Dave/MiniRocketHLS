#!/usr/bin/env python3
"""
fpga_network_host.py — Hardened in-data-path measurement host for the
fpga-network MiniRocket kernel (CMAC + NetLayer + fused MiniRocket).

Kernel contract (from fpga-network/minirocket/include/minirocket.hpp and
fpga-network/minirocket/src/minirocket.cpp):
  * 512-bit AXI-Stream pkt. Only the low 32 bits carry a float32 sample.
  * ONE pkt == one scalar time point. Kernel maintains a shift-register of
    length `time_series_length` and emits num_classes float32 logits on
    EVERY incoming sample (steady-state classification).
  * Output: num_classes float32 values, each in the low 32 bits of a 512-bit
    pkt, tlast=1 on every output pkt.
  * Model parameters (coefficients, biases, scaler, dilations, ...) are
    loaded via HBM on the FPGA side through a separate host step (xclbin
    program / PCIe prelude). This host ONLY streams samples over UDP and
    collects logits.

Wire format over UDP:
  Each UDP payload = one 64-byte (512-bit) word carrying a single float32 in
  bytes [0..4), zeros elsewhere. The NetLayer TLAST boundary is controlled on
  the FPGA side; we simply send a steady stream.

  NOTE: The actual NetLayer framing (header bytes, stream id, etc.) is
  defined by the 100G-fpga-network-stack-core IP. If the deployed NetLayer
  expects a non-trivial header (e.g. Xilinx XUP NetLayer appends its own),
  that is handled by the kernel-side NetLayer IP. This host emits the raw
  512-bit beat payload only; adjust HEADER/PAD if a wrapped transport is
  required — discovered at deploy time against the real node.

Outputs under results/networking/<dataset>_run<N>.json :
    dataset, series_length, num_classes, num_samples,
    throughput_inf_per_s, latency_us_p50/p95/p99, accuracy, wall_s,
    cmac_tx_bytes_before/after (if available), predictions[], errors.

Usage:
  python3 fpga_network_host.py \
      --dataset InsectSound --model-dir ../minirocket_modular \
      --fpga-ip 192.168.40.2 --fpga-port 62177 \
      --listen-port 62178 --runs 3 --out results/networking
"""
import argparse, json, os, socket, struct, sys, time, statistics, pathlib

BEAT_BYTES = 64  # 512-bit AXI-S beat

def load_dataset(model_dir, name):
    p = pathlib.Path(model_dir) / f"{name}_minirocket_model_test_data.json"
    with open(p) as f:
        d = json.load(f)
    import numpy as np
    X = np.asarray(d["X_test"], dtype=np.float32)
    y = np.asarray(d["y_test"], dtype=np.int32)
    return X, y, int(d["series_length"]), int(d["num_classes"])

def pack_beat(sample_f32: float) -> bytes:
    # float32 in low 4 bytes, zero pad to 64 B.
    return struct.pack("<f", sample_f32) + b"\x00" * (BEAT_BYTES - 4)

def unpack_logit(beat: bytes) -> float:
    return struct.unpack("<f", beat[:4])[0]

def run_one(X, y, series_length, num_classes, sock, fpga_addr, listen_sock):
    """Stream one full pass over X. Per-sample latency = time from sending
    the LAST beat of a time series to receiving the LAST logit for it."""
    n = X.shape[0]
    latencies_ns = []
    preds = [0]*n
    # Warmup: fill the shift register with the first TS (no timing).
    for t in range(series_length):
        sock.sendto(pack_beat(float(X[0, t])), fpga_addr)
    # Drain warmup logits (num_classes per beat * series_length beats).
    # In practice kernel only emits after shift-reg is full; we still drain.
    listen_sock.settimeout(2.0)
    drained = 0
    warmup_budget = series_length * num_classes
    try:
        while drained < warmup_budget:
            listen_sock.recvfrom(2048)
            drained += 1
    except socket.timeout:
        pass

    # Timed pass
    t_wall0 = time.perf_counter_ns()
    for i in range(n):
        # Send series_length beats for sample i
        t_send0 = time.perf_counter_ns()
        for t in range(series_length):
            sock.sendto(pack_beat(float(X[i, t])), fpga_addr)
        # Collect num_classes logits emitted for the LAST beat
        # (kernel emits num_classes logits per incoming beat; the logits
        # for the final beat of this TS are the prediction of interest).
        # We also drain the (series_length-1)*num_classes intermediate
        # logits produced by earlier beats of this TS.
        to_drain = (series_length - 1) * num_classes
        try:
            for _ in range(to_drain):
                listen_sock.recvfrom(2048)
            logits = [0.0]*num_classes
            for c in range(num_classes):
                data, _ = listen_sock.recvfrom(2048)
                logits[c] = unpack_logit(data)
            t_recv = time.perf_counter_ns()
            latencies_ns.append(t_recv - t_send0)
            preds[i] = max(range(num_classes), key=lambda k: logits[k])
        except socket.timeout:
            print(f"[WARN] timeout on sample {i}", file=sys.stderr)
            latencies_ns.append(-1)
    t_wall1 = time.perf_counter_ns()

    wall_s = (t_wall1 - t_wall0) / 1e9
    valid_lat = [l/1000.0 for l in latencies_ns if l > 0]  # µs
    valid_lat.sort()
    def pct(v, p):
        if not v: return float("nan")
        k = min(len(v)-1, int(round(p/100.0 * (len(v)-1))))
        return v[k]
    acc = sum(1 for i in range(n) if preds[i] == int(y[i])) / n
    return {
        "num_samples": n,
        "series_length": series_length,
        "num_classes": num_classes,
        "wall_s": wall_s,
        "throughput_inf_per_s": n / wall_s if wall_s > 0 else 0.0,
        "latency_us_p50": pct(valid_lat, 50),
        "latency_us_p95": pct(valid_lat, 95),
        "latency_us_p99": pct(valid_lat, 99),
        "latency_us_max": valid_lat[-1] if valid_lat else float("nan"),
        "timeouts": sum(1 for l in latencies_ns if l <= 0),
        "accuracy": acc,
        "predictions": preds,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["InsectSound","MosquitoSound","FruitFlies","GunPoint"])
    ap.add_argument("--model-dir", default="../minirocket_modular")
    ap.add_argument("--fpga-ip", required=True)
    ap.add_argument("--fpga-port", type=int, required=True)
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="results/networking")
    ap.add_argument("--limit", type=int, default=0,
                    help="limit number of samples (0 = all)")
    args = ap.parse_args()

    X, y, L, C = load_dataset(args.model_dir, args.dataset)
    if args.limit and args.limit < X.shape[0]:
        X, y = X[:args.limit], y[:args.limit]
    print(f"[INFO] {args.dataset}: N={X.shape[0]} L={L} C={C}")

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", args.listen_port))

    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in range(1, args.runs + 1):
        print(f"[INFO] run {r}/{args.runs}")
        res = run_one(X, y, L, C, tx, (args.fpga_ip, args.fpga_port), rx)
        res["dataset"] = args.dataset
        res["run"] = r
        res["fpga"] = f"{args.fpga_ip}:{args.fpga_port}"
        res["ts_unix"] = time.time()
        out_path = out_dir / f"{args.dataset}_run{r}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[OK ] run {r}: {res['throughput_inf_per_s']:.1f} inf/s  "
              f"p50={res['latency_us_p50']:.2f}µs p99={res['latency_us_p99']:.2f}µs  "
              f"acc={res['accuracy']*100:.2f}%")
        runs.append(res)

    tps = [r["throughput_inf_per_s"] for r in runs]
    p99 = [r["latency_us_p99"] for r in runs]
    accs = [r["accuracy"] for r in runs]
    summary = {
        "dataset": args.dataset,
        "runs": args.runs,
        "throughput_median": statistics.median(tps),
        "throughput_cv": (statistics.pstdev(tps)/statistics.mean(tps)) if len(tps)>1 and statistics.mean(tps)>0 else 0.0,
        "p99_worst_us": max(p99),
        "accuracy_median": statistics.median(accs),
    }
    with open(out_dir / f"{args.dataset}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] median={summary['throughput_median']:.1f} inf/s  "
          f"p99(worst)={summary['p99_worst_us']:.2f}µs  "
          f"acc={summary['accuracy_median']*100:.2f}%  "
          f"CV={summary['throughput_cv']*100:.2f}%")

if __name__ == "__main__":
    main()
