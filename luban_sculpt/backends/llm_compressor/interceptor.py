"""oneshot 前组装 / 改写 llm-compressor modifier 链。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from luban_sculpt.backends.llm_compressor.modifier_registry import resolve_modifier_class
from luban_sculpt.backends.llm_compressor.recipe_builder import build_base_modifiers
from luban_sculpt.contracts import BackendPlan

logger = logging.getLogger(__name__)


@dataclass
class ModifierInterceptor:
    """根据 recipe backend_options.modifiers 改写 llm-compressor modifier 链。"""

    plan: BackendPlan
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
        chain = list(base_modifiers if base_modifiers is not None else build_base_modifiers(self.plan))
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
        lc_mods = [m for m in modifiers if not isinstance(m, dict)]
        if not lc_mods:
            # dry-run stub：整链 dict 列表保留，便于 recipe_stub.json
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
