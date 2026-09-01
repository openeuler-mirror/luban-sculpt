"""Merged arch quant policies: MoE + dense."""

from __future__ import annotations

from luban_sculpt.model.dense import DENSE_PRESETS
from luban_sculpt.model.moe import MOE_PRESETS
from luban_sculpt.model.types import ArchQuantPolicy, ModelArch

ARCH_POLICIES: dict[ModelArch, ArchQuantPolicy] = {
    **DENSE_PRESETS,
    **MOE_PRESETS,
}


def get_arch_policy(arch: ModelArch) -> ArchQuantPolicy:
    return ARCH_POLICIES.get(arch, ARCH_POLICIES[ModelArch.UNKNOWN])
