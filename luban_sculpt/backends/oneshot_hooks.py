"""Shared oneshot lifecycle hooks for quant backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from luban_sculpt.contracts import BackendPlan
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


def run_pre_oneshot(plan: BackendPlan) -> Any:
    """oneshot 前：构建 LLMCompressorModifierManager 并记录 modifier 链。"""
    logger.info(
        "pre_oneshot backend=%s scheme=%s profile=%s",
        plan.intent.backend,
        plan.intent.abstract_scheme,
        plan.hw.profile_id,
    )
    from luban_sculpt.modifiers.recipe import LLMCompressorModifierManager

    manager = LLMCompressorModifierManager(plan)
    specs = manager.specs_from_plan()
    if specs:
        logger.info("modifier chain: %s", [s.get("name") for s in specs])
    return manager


def run_post_oneshot(plan: BackendPlan, output_dir: Path) -> None:
    """oneshot 后写入 ``luban_oneshot.json`` sidecar。"""
    logger.info("post_oneshot output=%s export=%s", output_dir, plan.export_format.value)
    sidecar = output_dir / "luban_oneshot.json"
    payload = {
        "backend": plan.intent.backend,
        "abstract_scheme": plan.intent.abstract_scheme,
        "profile_id": plan.hw.profile_id,
        "modifiers": (plan.intent.backend_options or {}).get("modifiers"),
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
