"""FP8 linear kernel selector stub (HAL layout / encoding from manifest)."""

from __future__ import annotations

from typing import Any


def domestic_fp8_linear(
    input: Any,
    weight: Any,
    scale: Any,
    *,
    encoding: str = "ieee_e4m3_ref",
    layout: str = "ct_pack",
) -> Any:
    raise NotImplementedError(
        f"domestic_fp8_linear stub (encoding={encoding}, layout={layout}); "
        "wire to NPU/DCU fused GEMM"
    )
