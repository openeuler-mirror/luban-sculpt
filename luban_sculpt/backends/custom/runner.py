from __future__ import annotations

from pathlib import Path

from luban_sculpt.backends.backend_util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact


class CustomCompressorBackend(QuantBackend):
    """用户自定义压缩流程占位：仅写 HAL manifest，具体量化由 recipe/options 外置脚本完成。"""

    name = "custom"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        opts = plan.intent.backend_options or {}
        meta = {"backend": "custom", "export": plan.export_format.value}
        if opts.get("custom"):
            meta["custom"] = opts["custom"]
        return run_with_hal(plan, Path(output_dir), meta)
