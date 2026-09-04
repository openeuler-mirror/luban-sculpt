"""Modifier 链：插件协议 + Manager 编排（与 pipeline recipe 无关）。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ChainModifier",
    "LubanChainHook",
    "WrapModifier",
    "LubanWrapModifier",
    "ModifierManager",
    "ModifierInterceptor",
    "LLMCompressorModifierManager",
    "build_recipe_for_plan",
    "build_base_modifiers",
    "list_modifier_names",
    "resolve_modifier_class",
    "discover_modifier_classes",
]


def __getattr__(name: str) -> Any:
    if name in ("ChainModifier", "LubanChainHook", "WrapModifier", "LubanWrapModifier"):
        from luban_sculpt.modifiers import base as _base

        return getattr(_base, name)
    if name == "ModifierManager":
        from luban_sculpt.modifiers.manager import ModifierManager

        return ModifierManager
    if name in ("ModifierInterceptor", "LLMCompressorModifierManager"):
        from luban_sculpt.modifiers.recipe import LLMCompressorModifierManager

        return LLMCompressorModifierManager
    if name == "build_recipe_for_plan":
        from luban_sculpt.modifiers.recipe import build_recipe_for_plan

        return build_recipe_for_plan
    if name == "build_base_modifiers":
        from luban_sculpt.modifiers.builders.llm_compressor import build_base_modifiers

        return build_base_modifiers
    if name in (
        "list_modifier_names",
        "resolve_modifier_class",
        "discover_modifier_classes",
    ):
        from luban_sculpt.modifiers import registry as _reg

        return getattr(_reg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
