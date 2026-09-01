"""llm-compressor Modifier 子类与 recipe 链拦截（本 backend 目录内聚）。"""

from __future__ import annotations

import logging
from typing import Any

from luban_sculpt.backends.llm_compressor._lc_import import (
    ensure_lc_modifier_classes,
    is_llmcompressor_available,
    probe_llmcompressor,
)
from luban_sculpt.contracts import BackendPlan

logger = logging.getLogger(__name__)


def _lc_available() -> bool:
    return is_llmcompressor_available()


class LubanChainHook:
    """Recipe YAML 驱动的链式 hook（不必继承 LC Modifier）。"""

    name: str = "LubanChainHook"

    def intercept(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        """按 spec.mode（append/prepend/replace/patch/wrap）插入或改写 modifier 链。"""
        mode = spec.get("mode", "append")
        built = self.build_lc_modifier(plan, spec)
        if built is None and mode != "patch":
            return self.patch_existing(modifiers, plan, spec)

        if mode == "patch":
            return self.patch_existing(modifiers, plan, spec)
        if mode == "prepend" and built is not None:
            return [built, *modifiers]
        if mode == "replace" and built is not None:
            return [built]
        if mode == "wrap" and built is not None and modifiers:
            return [LubanWrapModifier(built, modifiers[0]), *modifiers[1:]]
        if built is not None:
            return [*modifiers, built]
        return modifiers

    def build_lc_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        return None

    def patch_existing(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        """就地改写链上 QuantizationModifier 等。"""
        if not _lc_available():
            return modifiers
        QuantizationModifier = probe_llmcompressor()["QuantizationModifier"]
        for mod in modifiers:
            if isinstance(mod, QuantizationModifier):
                if spec.get("scheme"):
                    mod.scheme = spec["scheme"]
                if spec.get("targets"):
                    mod.targets = spec["targets"]
                if spec.get("ignore") is not None:
                    mod.ignore = spec["ignore"]
                elif plan.intent.ignore:
                    mod.ignore = list(plan.intent.ignore)
                if spec.get("block_size") is not None:
                    mod.block_size = spec["block_size"]
        return modifiers


class LubanWrapModifier:
    def __init__(self, outer: Any, inner: Any) -> None:
        self.outer = outer
        self.inner = inner

    def __repr__(self) -> str:
        return f"LubanWrapModifier({self.outer!r}, {self.inner!r})"


class LubanHALCalibModifier(LubanChainHook):
    """无 llmcompressor 时的 stub；有 LC 时由 HALCalibHook 换成真实 Modifier。"""

    name = "LubanHALCalibModifier"

    def build_lc_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        from luban_sculpt.hal.pipeline import HALPipeline

        return {
            "stub": "LubanHALCalibModifier",
            "calib_kernel": HALPipeline(hw=plan.hw).select_calib_kernel(),
        }


class LubanProducerMetadataModifier(LubanChainHook):
    name = "LubanProducerMetadataModifier"

    def build_lc_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        return {"stub": "LubanProducerMetadataModifier"}


class HALCalibHook(LubanChainHook):
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


class QuantizationPatch(LubanChainHook):
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


class DomesticFakeQuant(LubanChainHook):
    """追加一层 QuantizationModifier（scheme 可覆盖）。"""

    name = "DomesticFakeQuant"

    def build_lc_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        if not _lc_available():
            return {"stub": self.name, "profile": plan.hw.profile_id}
        QuantizationModifier = probe_llmcompressor()["QuantizationModifier"]
        return QuantizationModifier(
            targets=spec.get("targets", "Linear"),
            scheme=spec.get("scheme", "FP8_DYNAMIC"),
            ignore=spec.get("ignore") or plan.intent.ignore or ["lm_head"],
        )


class AscendFp8Block(LubanChainHook):
    name = "AscendFp8Block"

    def build_lc_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        if not _lc_available():
            return {"stub": self.name, "scheme": "FP8_BLOCK"}
        QuantizationModifier = probe_llmcompressor()["QuantizationModifier"]
        return QuantizationModifier(
            targets=spec.get("targets", "Linear"),
            scheme=spec.get("scheme", "FP8_BLOCK"),
            ignore=plan.intent.ignore or ["lm_head"],
            block_size=spec.get("block_size", 128),
        )


class ProducerMetadataHook(LubanChainHook):
    name = "LubanProducerMetadataModifier"

    def build_lc_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        lc = ensure_lc_modifier_classes()
        if not lc.get("available"):
            return {"stub": self.name}
        return lc["LubanProducerMetadataModifier"](
            producer={
                "name": "luban-sculpt",
                "profile_id": plan.hw.profile_id,
            }
        )


BUILTIN_LC_MODIFIERS: dict[str, type] = {
    "HALCalibHook": HALCalibHook,
    "LubanHALCalibModifier": HALCalibHook,
    "QuantizationPatch": QuantizationPatch,
    "DomesticFakeQuant": DomesticFakeQuant,
    "AscendFp8Block": AscendFp8Block,
    "LubanProducerMetadataModifier": ProducerMetadataHook,
}
# 兼容旧 registry 名称
DomesticFakeQuantModifier = DomesticFakeQuant
HALCalibHookModifier = HALCalibHook
AscendFp8BlockModifier = AscendFp8Block
