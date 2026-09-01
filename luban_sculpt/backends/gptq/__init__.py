from __future__ import annotations

from pathlib import Path

from luban_sculpt.backends._util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.backends.gptq.runner import run_gptqmodel_quantize
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact


class GPTQBackend(QuantBackend):
    """[ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) GPTQ 量化 → vLLM gptq。"""

    name = "gptq"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        """GPTQModel 量化并写 manifest（无 llm-compressor hooks）。"""
        out = Path(output_dir)
        gptq_meta = run_gptqmodel_quantize(plan, out)
        return run_with_hal(plan, out, gptq_meta)
