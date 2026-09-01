"""High-level: recipe file + profile id → BackendPlan."""

from __future__ import annotations

from pathlib import Path

from luban_sculpt.compiler.recipe_compiler import compile_recipe, load_recipe_yaml
from luban_sculpt.hae.engine import load_profile_template
from luban_sculpt.contracts import BackendPlan, HwDecision


def hw_for_profile(profile_name: str) -> HwDecision:
    """轻量 HwDecision（仅 profile_id/vendor），供 compile_plan 演示；完整 compress 走 HAE.run()。"""
    profile = load_profile_template(profile_name)
    return HwDecision(profile_id=profile_name, vendor=profile.get("vendor", "unknown"))


def compile_plan(recipe_path: Path | str, profile_name: str) -> BackendPlan:
    """从 recipe 文件与 profile_name 编译 BackendPlan（不跑完整 HAE 探测）。"""
    recipe = load_recipe_yaml(Path(recipe_path))
    profile = load_profile_template(profile_name)
    hw = hw_for_profile(profile_name)
    return compile_recipe(recipe, hw, profile)
