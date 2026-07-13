"""CPU MiniRocket baseline adapter.

Drives ``cpu_servers/cpu_minirocket_server.py`` over UDP. Wire format
(distinct from FPGA-axis):
    Request:  one UDP packet per sample =
              <u32 sample_id><float32 × series_length>
    Reply:    one UDP packet per prediction =
              <u32 sample_id><float32 × num_classes>

Sample IDs survive UDP reordering / drops, which lets the harness compute
drop_rate exactly as ``n_sent - n_received_unique``. We do NOT need FIFO
matching here, unlike the FPGA-axis adapter.

The adapter can either spawn the server as a child process (default, for
loopback smoke tests) or connect to an already-running server on a separate
host (for the apples-to-apples two-machine paper experiment).
"""
from __future__ import annotations

import os
import pathlib
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from math import isnan, nan
from typing import Dict, List, Optional, Tuple

from .base import AdapterConfig, InferenceAdapter, RunResult

DEFAULT_RECV_TIMEOUT_S = 0.05


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return nan
    k = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


class CpuMinirocketAdapter(InferenceAdapter):
    """Drive an external CPU MiniRocket UDP server.

    Construction args:
        model_json: path to the ``<dataset>_minirocket_model.json`` file
        dataset: dataset name (for RunResult.dataset / log keys)
        series_length / num_classes: must match the model
        server_host / server_port: where the CPU server is (or will be) bound
        listen_port: local port for replies (different from server_port)
        spawn_server: if True (default), spawn the server as a subprocess on
            ``server_host`` (must be ``127.0.0.1``/``localhost``). If False,
            assume the server is already running.
        server_python: python interpreter to use when spawning
    """

    platform = "cpu-minirocket"

    def __init__(
        self,
        model_json: str,
        dataset: str,
        series_length: int,
        num_classes: int,
        server_host: str = "127.0.0.1",
        server_port: int = 5050,
        listen_port: int = 5051,
        spawn_server: bool = True,
        server_python: Optional[str] = None,
        send_buf_bytes: int = 16 * 1024 * 1024,
        recv_buf_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.model_json = str(pathlib.Path(model_json).resolve())
        self.dataset = dataset
        self.series_length = int(series_length)
        self.num_classes = int(num_classes)
        self.server_addr: Tuple[str, int] = (server_host, int(server_port))
        self.listen_port = int(listen_port)
        self.spawn_server = bool(spawn_server)
        self.server_python = server_python or sys.executable
        self._send_buf_bytes = int(send_buf_bytes)
        self._recv_buf_bytes = int(recv_buf_bytes)

        self._server_proc: Optional[subprocess.Popen] = None
        self._tx: Optional[socket.socket] = None
        self._rx: Optional[socket.socket] = None
        self._req_fmt = "<I" + "f" * self.series_length
        self._req_size = struct.calcsize(self._req_fmt)
        self._rep_fmt = "<I" + "f" * self.num_classes
        self._rep_size = struct.calcsize(self._rep_fmt)

    # ---- lifecycle -------------------------------------------------------

    def warmup(self) -> None:
        if self.spawn_server and self._server_proc is None:
            self._server_proc = self._spawn_server()
        # Single socket for tx + rx so the server's reply (sent back to the
        # source address of the request) lands on the port the harness is
        # listening on. The previous design used separate sockets, which
        # made the server reply to an ephemeral port nobody was reading.
        if self._tx is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_buf_bytes)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_buf_bytes)
            sock.bind(("0.0.0.0", self.listen_port))
            sock.settimeout(DEFAULT_RECV_TIMEOUT_S)
            self._tx = sock
            self._rx = sock

    def _spawn_server(self) -> subprocess.Popen:
        ready_file = tempfile.NamedTemporaryFile(prefix="cpusrv_ready_", delete=False)
        ready_path = ready_file.name
        ready_file.close()
        os.unlink(ready_path)
        srv = pathlib.Path(__file__).resolve().parent.parent / "cpu_servers" / "cpu_minirocket_server.py"
        cmd = [
            self.server_python, str(srv),
            "--model-json", self.model_json,
            "--listen-ip", self.server_addr[0],
            "--listen-port", str(self.server_addr[1]),
            "--recv-buf-bytes", str(self._recv_buf_bytes),
            "--ready-file", ready_path,
        ]
        proc = subprocess.Popen(cmd)
        # Wait up to 30 s for ready file (model load + aeon import takes time).
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if os.path.exists(ready_path):
                break
            if proc.poll() is not None:
                raise RuntimeError(f"cpu-server exited early rc={proc.returncode}")
            time.sleep(0.05)
        else:
            proc.terminate()
            raise TimeoutError("cpu-server did not become ready within 30 s")
        try:
            os.unlink(ready_path)
        except OSError:
            pass
        return proc

    def close(self) -> None:
        # tx and rx share one socket; close it once.
        if self._tx is not None:
            try:
                self._tx.close()
            except OSError:
                pass
        self._tx = None
        self._rx = None
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    # ---- main run --------------------------------------------------------

    def run(self, samples: List, labels: Optional[List], cfg: AdapterConfig) -> RunResult:
        if self._tx is None:
            self.warmup()
        assert self._tx is not None and self._rx is not None

        rate = float(cfg.rate_pps)
        if rate <= 0.0:
            raise ValueError(f"rate_pps must be > 0, got {rate}")
        period_ns = int(1e9 / rate)

        # Pre-pack request payloads (without sample_id, which we set per send).
        # We use the dataset row index as sample_id; the same row may be sent
        # multiple times in one run, but the harness counts each *send* as a
        # distinct inference. To handle that, we use a monotonic uid that
        # wraps to u32, plus a side-table to recover the dataset row.
        n_dataset = len(samples)
        if n_dataset == 0:
            raise ValueError("empty samples")
        pre_packed_floats = [
            struct.pack("<" + "f" * self.series_length, *(float(v) for v in row))
            for row in samples
        ]

        warmup_ns = int(cfg.warmup_s * 1e9)
        run_ns = int(cfg.duration_s * 1e9)
        cooldown_ns = int(cfg.cooldown_s * 1e9)

        # Receiver state.
        recv_q: "queue.Queue[Tuple[int, bytes]]" = queue.Queue()
        stop_evt = threading.Event()

        def _receiver_loop() -> None:
            sock = self._rx
            assert sock is not None
            sock.settimeout(0.05)
            while not stop_evt.is_set():
                try:
                    data, _ = sock.recvfrom(self._rep_size + 64)
                except socket.timeout:
                    continue
                except OSError:
                    break
                recv_q.put((time.perf_counter_ns(), data))
            sock.settimeout(0.05)
            t_drain_start = time.perf_counter_ns()
            while time.perf_counter_ns() - t_drain_start < cooldown_ns:
                try:
                    data, _ = sock.recvfrom(self._rep_size + 64)
                except socket.timeout:
                    continue
                except OSError:
                    break
                recv_q.put((time.perf_counter_ns(), data))

        recv_thread = threading.Thread(target=_receiver_loop, name="cpu-rx", daemon=True)
        recv_thread.start()

        # Outgoing record indexed by uid (truncated to u32 for wire fit).
        # Maps uid -> (dataset_row, t_send_ns).
        sent: Dict[int, Tuple[int, int]] = {}

        t0_ns = time.perf_counter_ns()
        t_warmup_end_ns = t0_ns + warmup_ns
        t_run_end_ns = t_warmup_end_ns + run_ns

        # Pre-warmup: send a couple of samples to prime the server's caches /
        # JIT'd numpy paths. Discard their replies (will be gobbled by the
        # warmup window of the receiver thread).
        for i in range(min(2, n_dataset)):
            self._tx.sendto(struct.pack("<I", 0xFFFFFFFF) + pre_packed_floats[i], self.server_addr)

        uid_counter = 0
        next_deadline_ns = t_warmup_end_ns

        while True:
            now_ns = time.perf_counter_ns()
            if now_ns >= t_run_end_ns:
                break
            if now_ns < next_deadline_ns:
                slack_ns = next_deadline_ns - now_ns
                if slack_ns > 200_000:
                    time.sleep((slack_ns - 100_000) / 1e9)
                while time.perf_counter_ns() < next_deadline_ns:
                    pass
            row = uid_counter % n_dataset
            uid = uid_counter & 0xFFFFFFFF
            payload = struct.pack("<I", uid) + pre_packed_floats[row]
            t_send = time.perf_counter_ns()
            try:
                self._tx.sendto(payload, self.server_addr)
            except OSError as e:
                # Local socket buffer full. Count as a host-side drop.
                # Don't record it in `sent`; harness will see lower n_sent.
                _ = e
                next_deadline_ns += period_ns
                uid_counter += 1
                continue
            if t_send >= t_warmup_end_ns:
                sent[uid] = (row, t_send)
            uid_counter += 1
            next_deadline_ns += period_ns

        t_send_end_ns = time.perf_counter_ns()
        stop_evt.set()
        recv_thread.join(timeout=cfg.cooldown_s + 1.0)
        t_done_ns = time.perf_counter_ns()

        # Reduce replies. UID-based matching makes drops/reorders trivial.
        latencies_us: List[float] = []
        preds_by_uid: Dict[int, int] = {}
        recv_count = 0
        # Drain queue
        while True:
            try:
                t_recv_ns, data = recv_q.get_nowait()
            except queue.Empty:
                break
            if len(data) != self._rep_size:
                continue
            unpacked = struct.unpack(self._rep_fmt, data)
            uid = unpacked[0]
            if uid == 0xFFFFFFFF:
                continue  # warmup reply
            recv_count += 1
            if uid in sent:
                row, t_send_ns = sent[uid]
                latencies_us.append((t_recv_ns - t_send_ns) / 1000.0)
                logits = unpacked[1:]
                preds_by_uid[uid] = max(range(self.num_classes), key=lambda k: logits[k])

        latencies_us.sort()
        n_sent = len(sent)
        n_recv_matched = len(preds_by_uid)
        runtime_s = (t_send_end_ns - t_warmup_end_ns) / 1e9
        achieved = (n_recv_matched / runtime_s) if runtime_s > 0 else 0.0
        drop_count = max(0, n_sent - n_recv_matched)
        drop_rate = (drop_count / n_sent) if n_sent > 0 else 0.0

        acc = nan
        if labels is not None and n_recv_matched > 0:
            correct = 0
            for uid, pred in preds_by_uid.items():
                row, _ = sent[uid]
                if row < len(labels) and pred == int(labels[row]):
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
                "server_addr": f"{self.server_addr[0]}:{self.server_addr[1]}",
                "listen_port": self.listen_port,
                "spawn_server": self.spawn_server,
                "series_length": self.series_length,
                "num_classes": self.num_classes,
                "warmup_s": cfg.warmup_s,
                "cooldown_s": cfg.cooldown_s,
                "n_replies_total": recv_count,
                "drain_s": (t_done_ns - t_send_end_ns) / 1e9,
            },
        )
