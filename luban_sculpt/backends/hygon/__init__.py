from __future__ import annotations

from pathlib import Path

from luban_sculpt.backends._util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.backends.oneshot_hooks import run_post_oneshot, run_pre_oneshot
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


class HygonBackend(QuantBackend):
    """海光 DCU：llm-compressor 压缩 + enginex-hygon-vllm（slimquant / blockwise / awq_marlin）。"""

    name = "hygon"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        from luban_sculpt.backends.hygon.runner import run_hygon_quant

        logger.info(
            "hygon quantize output_dir=%s plan=%s",
            output_dir,
            plan.model_dump_json(indent=2),
        )
        run_pre_oneshot(plan)
        out = Path(output_dir)
        meta = run_hygon_quant(plan, out)
        artifact = run_with_hal(plan, out, meta)
        run_post_oneshot(plan, artifact.output_dir)
        return artifact
