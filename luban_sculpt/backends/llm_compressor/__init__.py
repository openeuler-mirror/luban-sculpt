"""llm-compressor backend：oneshot 执行 + scheme 映射。

Modifier 链编排见 ``luban_sculpt.modifiers``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from luban_sculpt.backends._util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact
from luban_sculpt.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "LLMCompressorBackend",
    "ModifierManager",
    "ModifierInterceptor",
    "LLMCompressorModifierManager",
    "build_recipe_for_plan",
    "list_modifier_names",
    "run_llm_compressor_oneshot",
    "is_llmcompressor_available",
    "resolve_compress_spec",
]


class LLMCompressorBackend(QuantBackend):
    """[vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) + Modifier 拦截。"""

    name = "llm_compressor"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        """pre hooks → oneshot → HAL/manifest → post hooks。"""
        from luban_sculpt.backends.llm_compressor.runner import run_llm_compressor_oneshot
        from luban_sculpt.backends.oneshot_hooks import run_post_oneshot, run_pre_oneshot

        logger.info(
            "llm_compressor quantize output_dir=%s plan=%s",
            output_dir,
            plan.model_dump_json(indent=2),
        )

        run_pre_oneshot(plan)
        out = Path(output_dir)
        lc_meta = run_llm_compressor_oneshot(plan, out)
        artifact = run_with_hal(plan, out, lc_meta)
        run_post_oneshot(plan, artifact.output_dir)
        return artifact


def __getattr__(name: str) -> Any:
    """Lazy re-exports so entry_point load does not import llmcompressor。"""
    if name in (
        "ModifierInterceptor",
        "ModifierManager",
        "LLMCompressorModifierManager",
    ):
        from luban_sculpt.modifiers.recipe import (
            LLMCompressorModifierManager,
            ModifierInterceptor,
        )

        if name == "LLMCompressorModifierManager":
            return LLMCompressorModifierManager
        return ModifierInterceptor if name == "ModifierInterceptor" else LLMCompressorModifierManager
    if name == "build_recipe_for_plan":
        from luban_sculpt.modifiers.recipe import build_recipe_for_plan

        return build_recipe_for_plan
    if name == "list_modifier_names":
        from luban_sculpt.modifiers.registry import list_modifier_names

        return list_modifier_names
    if name == "run_llm_compressor_oneshot":
        from luban_sculpt.backends.llm_compressor.runner import run_llm_compressor_oneshot

        return run_llm_compressor_oneshot
    if name == "is_llmcompressor_available":
        from luban_sculpt.backends.llm_compressor.probe import (
            is_llmcompressor_available,
        )

        return is_llmcompressor_available
    if name == "resolve_compress_spec":
        from luban_sculpt.backends.llm_compressor.scheme_map import resolve_compress_spec

        return resolve_compress_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
