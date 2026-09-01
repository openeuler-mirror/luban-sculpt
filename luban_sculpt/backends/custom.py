from __future__ import annotations

from pathlib import Path

from luban_sculpt.backends._util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact


class CustomCompressorBackend(QuantBackend):
    name = "custom"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        if plan.export_format.value not in ("compressed-tensors", "infera-v1"):
            raise ValueError("CustomCompressor must export CT or infera-v1")
        return run_with_hal(
            plan, Path(output_dir), {"backend": "custom", "export": plan.export_format.value}
        )
