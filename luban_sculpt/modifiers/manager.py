"""Modifier 链编排：base chain → 插件 apply_to_chain → finalize recipe。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from luban_sculpt.contracts import BackendPlan
from luban_sculpt.log import get_logger
from luban_sculpt.modifiers.registry import resolve_modifier_class

logger = get_logger(__name__)

BaseBuilder = Callable[[BackendPlan], list[Any]]
RecipeFinalizer = Callable[[list[Any]], Any]


def default_finalize_recipe(modifiers: list[Any]) -> Any:
    """默认 finalize：单元素直接返回，多元素尽量包成 llmcompressor Recipe。"""
    lc_mods = [m for m in modifiers if not isinstance(m, dict)]
    if not lc_mods:
        # dry-run stub：整链 dict 列表保留
        if len(modifiers) <= 1:
            return modifiers[0] if modifiers else None
        return modifiers
    if len(lc_mods) == 1:
        return lc_mods[0]
    try:
        from llmcompressor.recipe import Recipe

        return Recipe(modifiers=lc_mods)
    except Exception:
        return lc_mods


@dataclass
class ModifierManager:
    """根据 recipe backend_options.modifiers 改写 quant modifier 链。"""

    plan: BackendPlan
    base_builder: BaseBuilder | None = None
    finalizer: RecipeFinalizer | None = None
    log: list[str] = field(default_factory=list)

    def modifier_chain_specs(self) -> list[dict[str, Any]]:
        """Recipe ``compress.modifiers`` 编译进 plan 后的链配置（每项含 name/mode/参数）。

        读取 ``intent.backend_options["modifiers"]``（兼容旧键 ``modifier_chain``）。
        """
        opts = self.plan.intent.backend_options or {}
        raw = opts.get("modifiers") or opts.get("modifier_chain") or []
        if isinstance(raw, dict):
            return [raw]
        return list(raw)

    def apply(self, base_modifiers: list[Any] | None = None) -> list[Any]:
        """依次调用各 Modifier.apply_to_chain，合并为最终链。"""
        if base_modifiers is not None:
            chain = list(base_modifiers)
        elif self.base_builder is not None:
            chain = list(self.base_builder(self.plan))
        else:
            chain = []
        for spec in self.modifier_chain_specs():
            name = spec.get("name") or spec.get("modifier")
            if not name:
                continue
            cls = resolve_modifier_class(str(name))
            if cls is None:
                logger.warning("unknown modifier %s, skip", name)
                continue
            inst = cls()
            chain = inst.apply_to_chain(chain, self.plan, spec)
            self.log.append(f"apply_to_chain:{name}:mode={spec.get('mode', 'append')}")
        return chain

    def build_recipe(self) -> tuple[Any, list[Any], list[str]]:
        modifiers = self.apply()
        return self.finalize_recipe(modifiers), modifiers, self.log

    def finalize_recipe(self, modifiers: list[Any]) -> Any:
        fin = self.finalizer or default_finalize_recipe
        return fin(modifiers)


class LLMCompressorModifierManager(ModifierManager):
    """默认注入 llm-compressor base_builder / finalizer 的 Manager。"""

    def __init__(self, plan: BackendPlan, **kwargs: Any) -> None:
        from luban_sculpt.backends.llm_compressor.base_modifiers import (
            build_base_modifiers,
        )

        kwargs.setdefault("base_builder", build_base_modifiers)
        kwargs.setdefault("finalizer", default_finalize_recipe)
        super().__init__(plan, **kwargs)


def build_recipe_for_plan(plan: BackendPlan) -> tuple[Any, dict[str, Any]]:
    """ModifierManager 构建 llm-compressor Recipe，并返回拦截日志元数据。"""
    manager = LLMCompressorModifierManager(plan)
    recipe, modifiers, log = manager.build_recipe()
    return recipe, {
        "interceptor_log": log,
        "modifier_count": len(modifiers),
        "modifier_types": [type(m).__name__ for m in modifiers],
    }