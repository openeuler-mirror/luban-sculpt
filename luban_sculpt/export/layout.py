"""Optional weight repack after quant (FRACTAL_NZ, CT pack, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from luban_sculpt.contracts import HwDecision, WeightLayout


def maybe_repack(output_dir: Path, hw: HwDecision, weights_meta: dict[str, Any]) -> dict[str, Any]:
    layout = hw.weight_layout
    if layout == WeightLayout.FRACTAL_NZ:
        return {**weights_meta, "repack": "fractal_nz_stub"}
    if layout == WeightLayout.ROW_MAJOR_ALIGN:
        return {**weights_meta, "repack": "row_major_align_256_stub"}
    return weights_meta
