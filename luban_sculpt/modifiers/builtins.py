"""内置 ChainModifier（当前主要服务 llm-compressor 链）。"""

from __future__ import annotations

import logging
from typing import Any

from luban_sculpt.backends.llm_compressor.probe import (
    ensure_lc_modifier_classes,
    is_llmcompressor_available,
    probe_llmcompressor,
)
from luban_sculpt.contracts import BackendPlan
from luban_sculpt.modifiers.base import ChainModifier

logger = logging.getLogger(__name__)


def _lc_available() -> bool:
    return is_llmcompressor_available()


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
        lc = ensure_lc_modifier_classes()
        if lc.get("available") and mode in ("prepend", "append"):
            mod = lc["LubanHALCalibModifier"](
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
        if not _lc_available():
            return {"stub": self.name, "profile": plan.hw.profile_id}
        QuantizationModifier = probe_llmcompressor()["QuantizationModifier"]
        return QuantizationModifier(
            targets=spec.get("targets", "Linear"),
            scheme=spec.get("scheme", "FP8_DYNAMIC"),
            ignore=spec.get("ignore") or plan.intent.ignore or ["lm_head"],
        )


class AscendFp8Block(ChainModifier):
    name = "AscendFp8Block"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        if not _lc_available():
            return {"stub": self.name, "scheme": "FP8_BLOCK"}
        QuantizationModifier = probe_llmcompressor()["QuantizationModifier"]
        # FP8_BLOCK scheme 自带 block 语义；QuantizationModifier 不接受 block_size
        return QuantizationModifier(
            targets=spec.get("targets", "Linear"),
            scheme=spec.get("scheme", "FP8_BLOCK"),
            ignore=plan.intent.ignore or ["lm_head"],
        )


class ProducerMetadataHook(ChainModifier):
    name = "LubanProducerMetadataModifier"

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        lc = ensure_lc_modifier_classes()
        if not lc.get("available"):
            return {"stub": self.name}
        return lc["LubanProducerMetadataModifier"](
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
