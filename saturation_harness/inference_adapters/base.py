"""Adapter contract for the saturation harness.

Every platform under test (FPGA-MiniRocket-axis, FPGA-HYDRA-axis,
FPGA-MultiRocket-axis, CPU-MiniRocket, GPU-...) implements ``InferenceAdapter``.
``rate_sweep.py`` is platform-agnostic — it only knows about this interface.

Units convention (paper §4.5):
- Injection rate is in **inferences/sec** (a.k.a. samples/sec). One inference
  = one classification of a length-L time series. The wire-level beat rate is
  L * inf_per_sec for protocols that send one float per beat.
- Throughput is in **inferences/sec**.
- Latency is in **microseconds**, end-to-end (first beat sent → last logit
  for that sample received). Latency is only meaningful below saturation;
  above saturation, drops break per-sample matching, so latency fields may
  carry NaN.
- ``drop_count`` is the count of inferences not completed within the
  per-rate window. In Phase 1 this is approximated host-side as
  ``n_sent - n_received``; Phase 2 wires the FPGA AXI-Lite drop register.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AdapterConfig:
    """Adapter-agnostic configuration for one rate-sweep run.

    Adapter-specific knobs (FPGA IP, CPU server port, model file, ...) live on
    the concrete adapter constructor; this dataclass carries only the per-run
    parameters that ``rate_sweep.py`` itself sets.
    """
    rate_pps: float            # target inferences/sec
    duration_s: float          # how long to drive the platform at this rate
    warmup_s: float = 1.0      # discarded leading window
    cooldown_s: float = 0.5    # post-stop drain window for in-flight logits
    max_inflight: int = 0      # 0 = unbounded; >0 caps in-flight samples
    notes: str = ""


@dataclass
class RunResult:
    """One row of the throughput-vs-rate plot.

    Serialized as one line of JSONL into the run output directory.
    """
    # Configuration
    platform: str                          # "fpga-minirocket-axis" | "cpu-minirocket" | ...
    dataset: str                           # "GunPoint" | "InsectSound" | ...
    configured_rate_pps: float

    # Achieved
    achieved_throughput_inf_per_s: float
    n_samples_sent: int
    n_predictions_received: int
    drop_count: int                        # n_sent - n_received (Phase 1) or AXI-Lite reg (Phase 2)
    drop_rate: float                       # drop_count / n_samples_sent

    # Latency (only valid below saturation; can be NaN otherwise)
    latency_us_p50: float
    latency_us_p95: float
    latency_us_p99: float
    latency_us_max: float

    # Bookkeeping
    runtime_s: float
    accuracy: float                        # 0..1; NaN if labels unavailable
    ts_unix: float
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        import json
        return json.dumps(asdict(self), separators=(",", ":"))


class InferenceAdapter(ABC):
    """Pure interface; implementations live in this package."""

    #: Stable identifier used in JSON output and plot legends.
    platform: str = "abstract"

    #: Dataset under test. Adapters carry this so they can validate samples
    #: against their own model state.
    dataset: str = ""

    @abstractmethod
    def warmup(self) -> None:
        """Pre-run setup: load model, fill kernel shift register, etc."""

    @abstractmethod
    def run(self, samples: List, labels: Optional[List], cfg: AdapterConfig) -> RunResult:
        """Drive the platform at ``cfg.rate_pps`` for ``cfg.duration_s``.

        ``samples`` is a list of length-L float32 arrays (one per inference).
        The adapter cycles through ``samples`` if more inferences are needed
        than the dataset has rows.

        ``labels`` is optional ground truth. If provided, the adapter computes
        accuracy against received predictions; otherwise ``RunResult.accuracy``
        is NaN.
        """

    @abstractmethod
    def close(self) -> None:
        """Release sockets, OpenCL handles, subprocesses, etc."""

    def __enter__(self) -> "InferenceAdapter":
        self.warmup()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
