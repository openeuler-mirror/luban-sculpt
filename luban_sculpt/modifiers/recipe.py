"""Build full recipe: base modifiers + YAML modifier chain。"""

from __future__ import annotations

from typing import Any

from luban_sculpt.contracts import BackendPlan
from luban_sculpt.modifiers.builders.llm_compressor import build_base_modifiers
from luban_sculpt.modifiers.manager import ModifierManager, default_finalize_recipe


class LLMCompressorModifierManager(ModifierManager):
    """默认注入 llm-compressor base_builder / finalizer 的 Manager。"""

    def __init__(self, plan: BackendPlan, **kwargs: Any) -> None:
        kwargs.setdefault("base_builder", build_base_modifiers)
        kwargs.setdefault("finalizer", default_finalize_recipe)
        super().__init__(plan, **kwargs)


# 兼容旧名（曾位于 backends.llm_compressor.interceptor）
ModifierInterceptor = LLMCompressorModifierManager


def build_recipe_for_plan(plan: BackendPlan) -> tuple[Any, dict[str, Any]]:
    """ModifierManager 构建 llm-compressor Recipe，并返回拦截日志元数据。"""
    manager = LLMCompressorModifierManager(plan)
    recipe, modifiers, log = manager.build_recipe()
    return recipe, {
        "interceptor_log": log,
        "modifier_count": len(modifiers),
        "modifier_types": [type(m).__name__ for m in modifiers],
    }
