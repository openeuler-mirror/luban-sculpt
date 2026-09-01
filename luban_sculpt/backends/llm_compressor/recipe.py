"""Build full recipe: base QuantizationModifier + YAML modifier chain."""

from __future__ import annotations

from typing import Any

from luban_sculpt.backends.llm_compressor.interceptor import ModifierInterceptor
from luban_sculpt.contracts import BackendPlan


def build_recipe_for_plan(plan: BackendPlan) -> tuple[Any, dict[str, Any]]:
    """ModifierInterceptor 构建 llm-compressor Recipe，并返回拦截日志元数据。"""
    interceptor = ModifierInterceptor(plan)
    recipe, modifiers, log = interceptor.build_recipe()
    return recipe, {
        "interceptor_log": log,
        "modifier_count": len(modifiers),
        "modifier_types": [type(m).__name__ for m in modifiers],
    }
