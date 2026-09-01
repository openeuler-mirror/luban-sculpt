from __future__ import annotations

from pathlib import Path

from luban_sculpt.backends._util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact


class AWQBackend(QuantBackend):
    name = "awq"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        return run_with_hal(
            plan, Path(output_dir), {"backend": "awq", "export": plan.export_format.value}
        )
