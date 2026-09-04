"""Modifier 链编排：base chain → 插件 intercept → finalize recipe。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from luban_sculpt.contracts import BackendPlan
from luban_sculpt.modifiers.registry import resolve_modifier_class

logger = logging.getLogger(__name__)

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

    def specs_from_plan(self) -> list[dict[str, Any]]:
        """从 plan 解析 modifier 规格列表（name/mode/参数）。"""
        opts = self.plan.intent.backend_options or {}
        raw = opts.get("modifiers") or opts.get("modifier_chain") or []
        if isinstance(raw, dict):
            return [raw]
        return list(raw)

    def apply(self, base_modifiers: list[Any] | None = None) -> list[Any]:
        """依次调用各 Modifier.intercept，合并为最终链。"""
        if base_modifiers is not None:
            chain = list(base_modifiers)
        elif self.base_builder is not None:
            chain = list(self.base_builder(self.plan))
        else:
            chain = []
        for spec in self.specs_from_plan():
            name = spec.get("name") or spec.get("modifier")
            if not name:
                continue
            cls = resolve_modifier_class(str(name))
            if cls is None:
                logger.warning("unknown modifier %s, skip", name)
                continue
            inst = cls()
            chain = inst.intercept(chain, self.plan, spec)
            self.log.append(f"intercept:{name}:mode={spec.get('mode', 'append')}")
        return chain

    def build_recipe(self) -> tuple[Any, list[Any], list[str]]:
        modifiers = self.apply()
        return self.finalize_recipe(modifiers), modifiers, self.log

    def finalize_recipe(self, modifiers: list[Any]) -> Any:
        fin = self.finalizer or default_finalize_recipe
        return fin(modifiers)