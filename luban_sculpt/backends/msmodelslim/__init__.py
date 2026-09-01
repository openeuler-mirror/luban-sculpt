from __future__ import annotations

from pathlib import Path

from luban_sculpt.backends._util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.backends.msmodelslim.runner import run_msmodelslim_quant
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact
from luban_sculpt.backends.oneshot_hooks import run_post_oneshot, run_pre_oneshot


class MsModelSlimBackend(QuantBackend):
    """MindStudio ModelSlim — Ascend 一键量化，导出 vLLM-Ascend 权重。"""

    name = "msmodelslim"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        """msmodelslim CLI 量化 + HAL manifest + sidecar。"""
        run_pre_oneshot(plan)
        out = Path(output_dir)
        slim_meta = run_msmodelslim_quant(plan, out)
        artifact = run_with_hal(
            plan,
            out,
            {"backend": "msmodelslim", **slim_meta},
        )
        run_post_oneshot(plan, artifact.output_dir)
        return artifact
