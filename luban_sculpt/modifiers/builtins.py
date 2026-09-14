"""内置 ChainModifier（当前主要服务 llm-compressor 链）。"""

from __future__ import annotations

import logging
from typing import Any

from luban_sculpt.backends.llm_compressor.check import (
    ensure_llm_compressor_modifier_classes,
    is_llm_compressor_available,
    probe_llm_compressor,
)
from luban_sculpt.contracts import BackendPlan
from luban_sculpt.modifiers.base import ChainModifier

logger = logging.getLogger(__name__)


def _llm_compressor_available() -> bool:
    return is_llm_compressor_available()


class LubanHALCalibModifier(ChainModifier):
    """无 llmcompressor 时的 stub；有 LC 时由 HALCalibHook 换成真实 Modifier。"""

    name = "LubanHALCalibModifier"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        from luban_sculpt.hal.pipeline import HALPipeline

        return {
            "stub": "LubanHALCalibModifier",
            "calib_kernel": HALPipeline(hw=plan.hw).select_calib_kernel(),
        }


class LubanProducerMetadataModifier(ChainModifier):
    name = "LubanProducerMetadataModifier"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        return {"stub": "LubanProducerMetadataModifier"}


class HALCalibHook(ChainModifier):
    """兼容 recipe 名 HALCalibHook：prepend 真实 LC Modifier 或仅 patch。"""

    name = "HALCalibHook"

    def intercept(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        from luban_sculpt.hal.pipeline import HALPipeline

        kernel = HALPipeline(hw=plan.hw).select_calib_kernel()
        spec["_hal_calib_kernel"] = kernel
        mode = spec.get("mode", "prepend")
        probe_state = ensure_llm_compressor_modifier_classes()
        if probe_state.get("available") and mode in ("prepend", "append"):
            mod = probe_state["LubanHALCalibModifier"](
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

    def intercept(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        spec.setdefault("mode", "patch")
        return super().intercept(modifiers, plan, spec)


class DomesticFakeQuant(ChainModifier):
    """追加一层 QuantizationModifier（scheme 可覆盖）。"""

    name = "DomesticFakeQuant"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        if not _llm_compressor_available():
            return {"stub": self.name, "profile": plan.hw.profile_id}
        quantization_modifier = probe_llm_compressor()["QuantizationModifier"]
        return quantization_modifier(
            targets=spec.get("targets", "Linear"),
            scheme=spec.get("scheme", "FP8_DYNAMIC"),
            ignore=spec.get("ignore") or plan.intent.ignore or ["lm_head"],
        )


class AscendFp8Block(ChainModifier):
    name = "AscendFp8Block"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        if not _llm_compressor_available():
            return {"stub": self.name, "scheme": "FP8_BLOCK"}
        quantization_modifier = probe_llm_compressor()["QuantizationModifier"]
        # FP8_BLOCK scheme 自带 block 语义；QuantizationModifier 不接受 block_size
        return quantization_modifier(
            targets=spec.get("targets", "Linear"),
            scheme=spec.get("scheme", "FP8_BLOCK"),
            ignore=plan.intent.ignore or ["lm_head"],
        )


class ProducerMetadataHook(ChainModifier):
    name = "LubanProducerMetadataModifier"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        probe_state = ensure_llm_compressor_modifier_classes()
        if not probe_state.get("available"):
            return {"stub": self.name}
        return probe_state["LubanProducerMetadataModifier"](
            producer={
                "name": "luban-sculpt",
                "profile_id": plan.hw.profile_id,
            }
        )


BUILTIN_MODIFIERS: dict[str, type] = {
    "HALCalibHook": HALCalibHook,
    "LubanHALCalibModifier": HALCalibHook,
    "QuantizationPatch": QuantizationPatch,
    "DomesticFakeQuant": DomesticFakeQuant,
    "AscendFp8Block": AscendFp8Block,
    "LubanProducerMetadataModifier": ProducerMetadataHook,
}

# 兼容旧 registry 名称
BUILTIN_LC_MODIFIERS = BUILTIN_MODIFIERS
DomesticFakeQuantModifier = DomesticFakeQuant
HALCalibHookModifier = HALCalibHook
AscendFp8BlockModifier = AscendFp8Block
