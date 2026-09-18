"""Modifier 注册：entry_points + 内置 chain 插件。"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Type

ModifierClass = Type[Any]

_BUILTIN: dict[str, ModifierClass] | None = None


def _builtin_chain_modifiers() -> dict[str, ModifierClass]:
    global _BUILTIN
    if _BUILTIN is None:
        from luban_sculpt.modifiers.chain_modifiers import BUILTIN_CHAIN_MODIFIERS

        _BUILTIN = BUILTIN_CHAIN_MODIFIERS
    return _BUILTIN


def discover_modifier_classes() -> dict[str, ModifierClass]:
    mapping: dict[str, ModifierClass] = {}
    try:
        eps = entry_points(group="luban_sculpt.modifiers")
    except TypeError:
        eps = entry_points().get("luban_sculpt.modifiers", [])
    for ep in eps:
        mapping[ep.name] = ep.load()
    return mapping


def resolve_modifier_class(name: str) -> ModifierClass | None:
    discovered = discover_modifier_classes()
    if name in discovered:
        return discovered[name]
    return _builtin_chain_modifiers().get(name)


def list_modifier_names() -> list[str]:
    names = set(_builtin_chain_modifiers()) | set(discover_modifier_classes())
    return sorted(names)
