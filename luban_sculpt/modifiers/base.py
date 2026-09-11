"""Modifier 链插件协议（与具体 quant backend 解耦的编排层；patch 默认走 LC）。"""

from __future__ import annotations

from typing import Any

from luban_sculpt.contracts import BackendPlan


def _lc_available() -> bool:
    from luban_sculpt.backends.llm_compressor.probe import is_llmcompressor_available

    return is_llmcompressor_available()


class ChainModifier:
    """Recipe YAML 驱动的链式 hook（不必继承后端原生 Modifier）。"""

    name: str = "ChainModifier"

    def intercept(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        """按 spec.mode（append/prepend/replace/patch/wrap）插入或改写 modifier 链。"""
        mode = spec.get("mode", "append")
        # 优先走 build_lc_modifier，兼容只覆盖旧方法名的子类
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
            return [WrapModifier(built, modifiers[0]), *modifiers[1:]]
        if built is not None:
            return [*modifiers, built]
        return modifiers

    def build_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        """构建要插入链上的对象；默认无。"""
        return None

    # 兼容旧名
    def build_lc_modifier(self, plan: BackendPlan, spec: dict[str, Any]) -> Any | None:
        return self.build_modifier(plan, spec)

    def patch_existing(
        self,
        modifiers: list[Any],
        plan: BackendPlan,
        spec: dict[str, Any],
    ) -> list[Any]:
        """就地改写链上 QuantizationModifier 等（llm-compressor）。"""
        if not _lc_available():
            return modifiers
        from luban_sculpt.backends.llm_compressor.probe import probe_llmcompressor

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
                # block_size 仅 GPTQModifier 等支持；QuantizationModifier 会 pydantic 拒绝
                if spec.get("block_size") is not None and hasattr(mod, "block_size"):
                    try:
                        mod.block_size = spec["block_size"]
                    except (ValueError, TypeError, AttributeError):
                        pass
        return modifiers


# 兼容旧类名
LubanChainHook = ChainModifier


class WrapModifier:
    def __init__(self, outer: Any, inner: Any) -> None:
        self.outer = outer
        self.inner = inner

    def __repr__(self) -> str:
        return f"WrapModifier({self.outer!r}, {self.inner!r})"


# 兼容旧类名
LubanWrapModifier = WrapModifier
