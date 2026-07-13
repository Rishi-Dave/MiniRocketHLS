"""FPGA MiniRocket axis-stream adapter.

Drives the ``fpga-network/`` MiniRocket xclbin (CMAC + NetLayer + fused
MiniRocket) over UDP. Wire format matches ``cpu-network/fpga_network_host.py``:
each UDP datagram = one 64-byte (512-bit) AXI-S beat carrying one float32
sample in bytes [0..4), zero-padded thereafter. The kernel emits ``num_classes``
floats per incoming beat once the shift register is full.

Two threads:
- **Sender** paces sample emission with a deadline-driven token bucket
  (``time.perf_counter()``) to hit ``cfg.rate_pps`` inferences/sec.
- **Receiver** drains the listen socket into a queue and reconstructs
  per-sample predictions in FIFO order.

Drop accounting (Phase 1, no FPGA drop module yet):
- ``drop_count = n_sent - n_received`` after a ``cooldown_s`` drain.
- This is correct as long as kernel never reorders predictions, which holds
  per the AXI-S contract.
- Phase 2 replaces this with the FPGA AXI-Lite drop register read.

Latency caveat:
- Per-sample latency requires matching predictions back to send timestamps.
  At/above saturation, dropped beats break this matching for downstream
  samples. We therefore report latency only on the contiguous prefix of
  successfully-received predictions; if a drop is detected mid-run we tag
  the result with ``notes`` and may surface NaN latencies.
"""
from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from math import isnan, nan
from typing import List, Optional, Tuple

from .base import AdapterConfig, InferenceAdapter, RunResult

BEAT_BYTES = 64                # 512-bit AXI-S beat
FLOAT_PREFIX = 4               # bytes [0:4) carry float32; rest zero-pad
RECV_BUFSZ = 2048              # generous; payload is 64 B
DEFAULT_RECV_TIMEOUT_S = 1.0   # per-recvfrom; receiver thread loops

# Per-beat pacing cap (saturation_pivot 2026-05-06 Bug C). A tight
# `for beat in beats: sendto(beat)` produces ~1.5M pps bursts that overrun
# NetLayer's eth_in FIFO and stick the 100G CMAC GT in an unrecoverable
# state — only `xbutil reset --user` clears it. Cap per-beat rate to leave
# the CMAC RX path room to drain. Tunable via env var for sweeping.
DEFAULT_BEAT_PACE_NS = int(os.environ.get("MR_AXIS_BEAT_PACE_NS", "5000"))


def _pack_beat(x: float) -> bytes:
    return struct.pack("<f", x) + b"\x00" * (BEAT_BYTES - FLOAT_PREFIX)


def _unpack_logit(b: bytes) -> float:
    return struct.unpack("<f", b[:FLOAT_PREFIX])[0]


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return nan
    k = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


@dataclass
class _SendRecord:
    sample_idx: int            # row in the dataset
    t_first_send_ns: int       # wall clock of first beat sent
    t_last_send_ns: int        # wall clock of last beat sent


class MinirocketAxisAdapter(InferenceAdapter):
    """Drive the FPGA MiniRocket-axis kernel over UDP.

    Construction args:
        fpga_ip / fpga_port: where the FPGA NetLayer is listening
        listen_port: local UDP port the FPGA sends logits back to
        series_length / num_classes: kernel shape (must match xclbin model)
        dataset: name (used in RunResult.dataset)
        send_buf_bytes: SO_SNDBUF for the tx socket (raised to absorb bursts)
        recv_buf_bytes: SO_RCVBUF for the rx socket (raised to reduce host-side drops)
    """

    platform = "fpga-minirocket-axis"

    def __init__(
        self,
        fpga_ip: str,
        fpga_port: int,
        listen_port: int,
        series_length: int,
        num_classes: int,
        dataset: str,
        send_buf_bytes: int = 16 * 1024 * 1024,
        recv_buf_bytes: int = 64 * 1024 * 1024,
        beat_pace_ns: int = DEFAULT_BEAT_PACE_NS,
    ) -> None:
        self.fpga_addr: Tuple[str, int] = (fpga_ip, fpga_port)
        self.listen_port = listen_port
        self.series_length = int(series_length)
        self.num_classes = int(num_classes)
        self.dataset = dataset
        self._send_buf_bytes = int(send_buf_bytes)
        self._recv_buf_bytes = int(recv_buf_bytes)
        self._beat_pace_ns = int(beat_pace_ns)

        self._tx: Optional[socket.socket] = None
        self._rx: Optional[socket.socket] = None
        self._warmed_up = False

    def _send_beats_paced(self, beats: List[bytes], start_ns: int) -> int:
        """Send pre-packed beats with per-beat deadline pacing.

        Caps the rate to ~1/beat_pace_ns pps to avoid overrunning NetLayer's
        eth_in FIFO (saturation_pivot Bug C). Returns the perf_counter_ns
        timestamp after the final sendto.
        """
        sock = self._tx
        addr = self.fpga_addr
        period = self._beat_pace_ns
        next_ns = start_ns
        for beat in beats:
            while True:
                now = time.perf_counter_ns()
                if now >= next_ns:
                    break
                slack = next_ns - now
                if slack > 50_000:
                    time.sleep((slack - 50_000) / 1e9)
            sock.sendto(beat, addr)
            next_ns += period
        return time.perf_counter_ns()

    # ---- lifecycle -------------------------------------------------------

    def warmup(self) -> None:
        # Single socket bound to listen_port for BOTH tx and rx.
        # NetLayer's UDP RX SocketTable matches incoming packets on the
        # tuple (theirIP, theirPort, myPort); slot[0].theirPort is set to
        # `listen_port` by setup_netlayer. If our tx socket sourced from an
        # ephemeral port (the previous behaviour), 99.94% of harness packets
        # missed the SocketTable lookup and were silently dropped on-FPGA
        # (saturation_pivot 2026-05-05 NetLayer-counter diagnosis).
        # Sharing a single socket guarantees src_port == listen_port.
        if self._tx is None or self._rx is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_buf_bytes)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_buf_bytes)
            sock.bind(("0.0.0.0", self.listen_port))
            sock.settimeout(DEFAULT_RECV_TIMEOUT_S)
            self._tx = sock
            self._rx = sock
        self._warmed_up = True

    def close(self) -> None:
        # _tx and _rx now alias the same socket; close it once.
        sock = self._tx
        self._tx = None
        self._rx = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self._warmed_up = False

    # ---- main run --------------------------------------------------------

    def run(self, samples: List, labels: Optional[List], cfg: AdapterConfig) -> RunResult:
        if not self._warmed_up:
            self.warmup()
        assert self._tx is not None and self._rx is not None

        # Pre-pack beats per sample to take serialization out of the timed
        # loop. samples[i] is a length-L iterable of floats.
        packed = [
            [_pack_beat(float(x)) for x in row]
            for row in samples
        ]
        n_dataset = len(packed)
        if n_dataset == 0:
            raise ValueError("empty samples")

        rate = float(cfg.rate_pps)
        if rate <= 0.0:
            raise ValueError(f"rate_pps must be > 0, got {rate}")
        period_ns = int(1e9 / rate)

        # Plan: drive for cfg.duration_s + cfg.warmup_s; discard warmup window.
        warmup_ns = int(cfg.warmup_s * 1e9)
        run_ns = int(cfg.duration_s * 1e9)
        cooldown_ns = int(cfg.cooldown_s * 1e9)

        # Receiver state. We push every received beat into a queue with its
        # arrival timestamp; the post-run reducer walks the queue in FIFO
        # order and groups every num_classes beats into one prediction.
        recv_q: "queue.Queue[Tuple[int, bytes]]" = queue.Queue()
        stop_evt = threading.Event()

        def _receiver_loop() -> None:
            sock = self._rx
            assert sock is not None
            sock.settimeout(0.05)
            while not stop_evt.is_set():
                try:
                    data, _ = sock.recvfrom(RECV_BUFSZ)
                except socket.timeout:
                    continue
                except OSError:
                    break
                recv_q.put((time.perf_counter_ns(), data))
            # Drain anything left in kernel buffers post-stop (cooldown).
            sock.settimeout(0.05)
            t_drain_start = time.perf_counter_ns()
            while time.perf_counter_ns() - t_drain_start < cooldown_ns:
                try:
                    data, _ = sock.recvfrom(RECV_BUFSZ)
                except socket.timeout:
                    continue
                except OSError:
                    break
                recv_q.put((time.perf_counter_ns(), data))

        recv_thread = threading.Thread(target=_receiver_loop, name="rx", daemon=True)
        recv_thread.start()

        send_records: List[_SendRecord] = []
        t0_ns = time.perf_counter_ns()
        t_warmup_end_ns = t0_ns + warmup_ns
        t_run_end_ns = t_warmup_end_ns + run_ns

        # Pre-warmup fill: send one full series with index 0 to populate the
        # kernel shift register. Discard whatever logits come back during the
        # warmup window. Paced to avoid wedging the FPGA's CMAC RX.
        self._send_beats_paced(packed[0], time.perf_counter_ns())

        sample_counter = 0
        next_deadline_ns = t_warmup_end_ns  # first paced send aligns with warmup end

        # Sender hot loop. Deadline-driven; busy-wait when close so we avoid
        # OS sleep granularity (~1 ms) which would distort high-rate runs.
        while True:
            now_ns = time.perf_counter_ns()
            if now_ns >= t_run_end_ns:
                break
            # Wait until the next deadline.
            if now_ns < next_deadline_ns:
                # Sleep coarsely if we have headroom, then spin.
                slack_ns = next_deadline_ns - now_ns
                if slack_ns > 200_000:  # >0.2 ms
                    time.sleep((slack_ns - 100_000) / 1e9)
                while time.perf_counter_ns() < next_deadline_ns:
                    pass
            # Send one sample's worth of beats with per-beat pacing (Bug C).
            idx = sample_counter % n_dataset
            beats = packed[idx]
            t_first = time.perf_counter_ns()
            t_last = self._send_beats_paced(beats, t_first)
            # Only record samples sent inside the timed window.
            if t_last >= t_warmup_end_ns:
                send_records.append(_SendRecord(idx, t_first, t_last))
            sample_counter += 1
            next_deadline_ns += period_ns

        t_send_end_ns = time.perf_counter_ns()
        # Stop receiver and wait for cooldown drain.
        stop_evt.set()
        recv_thread.join(timeout=cfg.cooldown_s + 1.0)
        t_done_ns = time.perf_counter_ns()

        # Reduce received beats into predictions.
        n_classes = self.num_classes
        L = self.series_length
        # v5 wire format (2026-05-07): kernel emits ONE packet per input beat
        # once the shift register is full, with num_classes float32 logits
        # packed at the start of the 64-byte payload. Per inference we send
        # L input beats and expect L response packets back; the LAST packet
        # of each L-window carries the final prediction (the one computed
        # once the shift register holds the complete new series).
        beats: List[Tuple[int, bytes]] = []
        while True:
            try:
                beats.append(recv_q.get_nowait())
            except queue.Empty:
                break

        per_sample_packets = L
        n_predictions_received = len(beats) // per_sample_packets

        # Match predictions to send records FIFO. Drop accounting:
        # n_sent (in window) vs n_received.
        n_sent = len(send_records)
        n_recv_matched = min(n_predictions_received, n_sent)
        latencies_us: List[float] = []
        preds: List[int] = []
        for i in range(n_recv_matched):
            # Last packet of the L-window carries the final prediction logits.
            last_pkt_idx = (i + 1) * per_sample_packets - 1
            t_recv_ns, payload = beats[last_pkt_idx]
            logits = [
                struct.unpack("<f", payload[c * 4 : c * 4 + 4])[0]
                for c in range(n_classes)
            ]
            preds.append(max(range(n_classes), key=lambda k: logits[k]))
            t_first_send = send_records[i].t_first_send_ns
            latencies_us.append((t_recv_ns - t_first_send) / 1000.0)

        latencies_us.sort()
        runtime_s = (t_send_end_ns - t_warmup_end_ns) / 1e9
        achieved = (n_recv_matched / runtime_s) if runtime_s > 0 else 0.0
        drop_count = max(0, n_sent - n_recv_matched)
        drop_rate = (drop_count / n_sent) if n_sent > 0 else 0.0

        # Accuracy (only over matched preds, since unmatched are missing).
        acc = nan
        if labels is not None and n_recv_matched > 0:
            correct = 0
            for i in range(n_recv_matched):
                idx = send_records[i].sample_idx
                if idx < len(labels) and preds[i] == int(labels[idx]):
                    correct += 1
            acc = correct / n_recv_matched

        notes = []
        if drop_rate > 0.001:
            notes.append(f"host-observed drop_rate={drop_rate:.3f}")
        if isnan(acc):
            notes.append("no labels supplied")
        return RunResult(
            platform=self.platform,
            dataset=self.dataset,
            configured_rate_pps=rate,
            achieved_throughput_inf_per_s=achieved,
            n_samples_sent=n_sent,
            n_predictions_received=n_recv_matched,
            drop_count=drop_count,
            drop_rate=drop_rate,
            latency_us_p50=_percentile(latencies_us, 50),
            latency_us_p95=_percentile(latencies_us, 95),
            latency_us_p99=_percentile(latencies_us, 99),
            latency_us_max=latencies_us[-1] if latencies_us else nan,
            runtime_s=runtime_s,
            accuracy=acc,
            ts_unix=time.time(),
            notes="; ".join(notes),
            extra={
                "fpga_addr": f"{self.fpga_addr[0]}:{self.fpga_addr[1]}",
                "listen_port": self.listen_port,
                "series_length": L,
                "num_classes": n_classes,
                "warmup_s": cfg.warmup_s,
                "cooldown_s": cfg.cooldown_s,
                "n_beats_received": len(beats),
                "drain_s": (t_done_ns - t_send_end_ns) / 1e9,
            },
        )
