"""High-level: recipe file + profile id → BackendPlan."""

from __future__ import annotations

from pathlib import Path

from luban_sculpt.compiler.recipe_compiler import compile_recipe, load_recipe_yaml
from luban_sculpt.hae.engine import load_profile_template
from luban_sculpt.contracts import BackendPlan, HwDecision
from luban_sculpt.pipeline.config import build_stage_recipe, parse_pipeline_config


def hw_for_profile(profile_name: str) -> HwDecision:
    """轻量 HwDecision（仅 profile_id/vendor），供 compile_plan 演示；完整 compress 走 HAE.run()。"""
    profile = load_profile_template(profile_name)
    return HwDecision(profile_id=profile_name, vendor=profile.get("vendor", "unknown"))


def compile_plan(recipe_path: Path | str, profile_name: str) -> BackendPlan:
    """从 recipe 文件与 profile_name 编译 BackendPlan（不跑完整 HAE 探测）。"""
    recipe = load_recipe_yaml(Path(recipe_path))
    profile = load_profile_template(profile_name)
    hw = hw_for_profile(profile_name)
    pipeline = parse_pipeline_config(recipe)
    stages = pipeline.enabled_stages()
    if not stages:
        raise ValueError(f"{recipe_path}: pipeline has no enabled stages")
    stage = stages[0]
    stage_recipe = build_stage_recipe(
        recipe, stage, model_id=str(recipe["model_id"])
    )
    return compile_recipe(stage_recipe, hw, profile)
