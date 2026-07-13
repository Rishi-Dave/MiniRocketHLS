"""FPGA MultiRocket axis-stream adapter.

Drives the ``multirocket_optimized/multirocket_axis/build_dir.hw/krnl.xclbin``
over UDP. Wire format identical to MiniRocket-axis. MultiRocket has a much
larger feature vector (~5,376 features vs. MiniRocket's ~9,996 PPV-only
features × 1 pooling = ~10k vs MultiRocket's 84 × dilations × 4 pooling × 2
representations) but that's an HBM concern, not a wire concern — the host
sees the same one-float-per-beat protocol.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

from .minirocket_axis_adapter import MinirocketAxisAdapter


def load_multirocket_model_meta(model_json_path: str):
    """Return (series_length, num_classes) from a MultiRocket model JSON.

    Schema (post Jan-6 fix): ``time_series_length`` and ``num_classes`` are
    top-level keys, same as MiniRocket / HYDRA model JSON.
    """
    with open(model_json_path) as f:
        m = json.load(f)
    return int(m["time_series_length"]), int(m["num_classes"])


class MultirocketAxisAdapter(MinirocketAxisAdapter):
    """MultiRocket-axis FPGA adapter."""

    platform = "fpga-multirocket-axis"

    def __init__(
        self,
        fpga_ip: str,
        fpga_port: int,
        listen_port: int,
        series_length: int,
        num_classes: int,
        dataset: str,
        **kw,
    ) -> None:
        super().__init__(
            fpga_ip=fpga_ip,
            fpga_port=fpga_port,
            listen_port=listen_port,
            series_length=series_length,
            num_classes=num_classes,
            dataset=dataset,
            **kw,
        )

    @classmethod
    def from_model_json(
        cls,
        model_json: str,
        fpga_ip: str,
        fpga_port: int,
        listen_port: int,
        dataset: Optional[str] = None,
        **kw,
    ) -> "MultirocketAxisAdapter":
        L, C = load_multirocket_model_meta(model_json)
        ds = dataset or pathlib.Path(model_json).stem.replace("multirocket84_", "").replace("_model", "")
        return cls(
            fpga_ip=fpga_ip,
            fpga_port=fpga_port,
            listen_port=listen_port,
            series_length=L,
            num_classes=C,
            dataset=ds,
            **kw,
        )
