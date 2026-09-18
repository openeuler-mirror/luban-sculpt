"""Modifier 链：插件协议 + Manager 编排（llm-compressor recipe 构建）。"""

from __future__ import annotations

from typing import Any

# 对外入口（懒加载，见 __getattr__）
__all__ = [
    "ChainModifier",
    "WrapModifier",
    "ModifierManager",
    "LLMCompressorModifierManager",
    "build_recipe_for_plan",
    "list_modifier_names",
    "resolve_modifier_class",
]


def __getattr__(name: str) -> Any:
    if name in ("ChainModifier", "WrapModifier"):
        from luban_sculpt.modifiers import base as _base

        return getattr(_base, name)
    if name == "ModifierManager":
        from luban_sculpt.modifiers.manager import ModifierManager

        return ModifierManager
    if name in ("LLMCompressorModifierManager", "build_recipe_for_plan"):
        from luban_sculpt.modifiers import manager as _mgr

        return getattr(_mgr, name)
    if name in ("list_modifier_names", "resolve_modifier_class"):
        from luban_sculpt.modifiers import registry as _reg

        return getattr(_reg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
