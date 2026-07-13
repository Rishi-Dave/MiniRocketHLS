"""FPGA HYDRA axis-stream adapter.

Drives the ``hydra_optimized/hydra_axis/build_dir.hw/krnl.xclbin`` over UDP.
Wire format is identical to MiniRocket axis (one float per 64-byte 512-bit
beat, kernel emits ``num_classes`` floats per incoming beat once the shift
register is primed). The only differences vs. ``minirocket_axis_adapter``
are:

- Different ``platform`` tag for plot legends.
- Loads HYDRA model JSON (different schema: ``kernel_weights`` flat 512×9,
  ``group_assignments``, ``num_groups``, etc.) instead of MiniRocket's
  ``kernel_indices`` schema. We don't actually need to parse the model
  here — model parameters are pre-loaded into HBM by the host setup phase
  before the harness streams samples — but we read it to extract
  ``time_series_length`` and ``num_classes`` for shape sanity.
- Default UDP ports differ from MiniRocket so both can run side-by-side
  during multi-platform sweeps.

For Phase 6 saturation experiments, instantiate with the FPGA IP / ports
configured at deployment time.
"""
from __future__ import annotations

import json
import pathlib
from typing import List, Optional

from .base import AdapterConfig, RunResult
from .minirocket_axis_adapter import MinirocketAxisAdapter


def load_hydra_model_meta(model_json_path: str):
    """Return (series_length, num_classes) from a HYDRA model JSON file."""
    with open(model_json_path) as f:
        m = json.load(f)
    return int(m["time_series_length"]), int(m["num_classes"])


class HydraAxisAdapter(MinirocketAxisAdapter):
    """HYDRA-axis FPGA adapter. UDP wire format inherited from MiniRocket-axis."""

    platform = "fpga-hydra-axis"

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
    ) -> "HydraAxisAdapter":
        L, C = load_hydra_model_meta(model_json)
        ds = dataset or pathlib.Path(model_json).stem.replace("hydra_", "").replace("_model", "")
        return cls(
            fpga_ip=fpga_ip,
            fpga_port=fpga_port,
            listen_port=listen_port,
            series_length=L,
            num_classes=C,
            dataset=ds,
            **kw,
        )
