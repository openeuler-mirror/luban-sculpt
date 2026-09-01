"""Modifier 注册：entry_points + 本目录 builtins。"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Type

from luban_sculpt.backends.llm_compressor.modifiers import BUILTIN_LC_MODIFIERS

ModifierClass = Type[Any]


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
    return BUILTIN_LC_MODIFIERS.get(name)


def list_modifier_names() -> list[str]:
    names = set(BUILTIN_LC_MODIFIERS) | set(discover_modifier_classes())
    return sorted(names)
