"""Shared wire-format + stats-reporting helpers for the CPU streaming-inference
UDP responders (minirocket_udp_responder.py, hydra_udp_responder.py).

Wire contract (must match the FPGA exactly -- see
MiniRocketHLS/cpu-network/fpga_network_host.py header comment and
MiniRocketHLS/fpga-network/minirocket_dpr/src/minirocket.cpp:283-297):

    1 UDP packet = 1 sample.
    Payload = 64 bytes total.
      bytes [0:4)  -> little-endian float32 sample value
      bytes [4:64) -> zero padding
    (This is one 512-bit AXI-Stream "beat" mirrored onto the wire.)

Note: MiniRocketHLS/fpga-network/network_experiments/TP/throughput.py (the
sweep driver these responders are meant to be driven by) sends a fixed
64-byte all-zero payload for pure rate/loss testing -- that unpacks to
sample=0.0, which is a perfectly valid (if degenerate) sample under this
wire contract, so no responder changes are needed to work with it.
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
from typing import Any, Dict

BEAT_BYTES = 64
SAMPLE_STRUCT = struct.Struct("<f")


def unpack_sample(beat: bytes) -> float:
    """Extract the float32 sample from bytes [0:4) of a 64B beat payload."""
    return SAMPLE_STRUCT.unpack_from(beat, 0)[0]


def pack_beat(value: float) -> bytes:
    """Pack a float32 into a 64B beat payload (low 4 bytes, zero-padded)."""
    return SAMPLE_STRUCT.pack(float(value)) + b"\x00" * (BEAT_BYTES - 4)


class StatsReporter:
    """Background thread that periodically snapshots a counters dict to a
    JSON file (atomic replace) so an external sweep-runner script can sample
    "packets actually processed" before/after each rate step without adding
    any per-packet I/O overhead to the hot recv loop.

    The hot loop only ever does plain dict-key increments under a lock;
    this thread does all the (comparatively expensive) file I/O.
    """

    def __init__(self, path: str, counters: Dict[str, Any], lock: threading.Lock,
                 interval_s: float = 0.2):
        self.path = path
        self.counters = counters
        self.lock = lock
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.interval_s)
            self._write_once()
        # final snapshot on shutdown
        self._write_once()

    def _write_once(self) -> None:
        with self.lock:
            snap = dict(self.counters)
        snap["ts_unix"] = time.time()
        tmp = f"{self.path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, self.path)
        except OSError:
            pass


def read_stats(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
