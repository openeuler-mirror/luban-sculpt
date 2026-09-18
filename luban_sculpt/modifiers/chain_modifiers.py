"""内置 ChainModifier 插件（YAML ``compress.modifiers`` → ``registry`` 解析）。"""

from __future__ import annotations

from typing import Any

from luban_sculpt.backends.llm_compressor.check import (
    get_llm_compressor_quantization_modifier_class,
    register_luban_llm_compressor_modifiers,
)
from luban_sculpt.contracts import BackendPlan
from luban_sculpt.log import get_logger
from luban_sculpt.modifiers.base import ChainModifier

logger = get_logger(__name__)


class HALCalibHook(ChainModifier):
    """Recipe 名 HALCalibHook：prepend/append 真实 LC LubanHALCalibModifier。"""

    name = "HALCalibHook"

    def apply_to_chain(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        from luban_sculpt.hal.pipeline import HALPipeline

        kernel = HALPipeline(hw=plan.hw).select_calib_kernel()
        spec["_hal_calib_kernel"] = kernel
        mode = spec.get("mode", "prepend")
        lc_state = register_luban_llm_compressor_modifiers()
        if lc_state.get("available") and mode in ("prepend", "append"):
            mod = lc_state["LubanHALCalibModifier"](
                calib_kernel=kernel,
                profile_name=plan.hw.profile_id,
            )
            if mode == "prepend":
                return [mod, *modifiers]
            return [*modifiers, mod]
        return modifiers


class QuantizationPatch(ChainModifier):
    """patch 模式：改链上 QuantizationModifier 的 scheme/ignore/block_size。"""

    name = "QuantizationPatch"

    def apply_to_chain(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        spec.setdefault("mode", "patch")
        return super().apply_to_chain(modifiers, plan, spec)


class DomesticFakeQuant(ChainModifier):
    """追加一层 QuantizationModifier（scheme 可覆盖）。"""

    name = "DomesticFakeQuant"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        quant_mod_cls = get_llm_compressor_quantization_modifier_class()
        if quant_mod_cls is None:
            return {"stub": self.name, "profile": plan.hw.profile_id}
        return quant_mod_cls(
            targets=spec.get("targets", "Linear"),
            scheme=spec.get("scheme", "FP8_DYNAMIC"),
            ignore=spec.get("ignore") or plan.intent.ignore or ["lm_head"],
        )


BUILTIN_CHAIN_MODIFIERS: dict[str, type] = {
    "HALCalibHook": HALCalibHook,
    "QuantizationPatch": QuantizationPatch,
    "DomesticFakeQuant": DomesticFakeQuant,
}
