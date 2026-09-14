"""CLI recipe override helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml
from luban_sculpt.pipeline.recipe_overrides import apply_recipe_cli_overrides
from tests.paths import LLAMA3_EXAMPLE


def _compile_doc(doc: dict, profile: str):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.safe_dump(doc, f)
        path = Path(f.name)
    try:
        return compile_plan(path, profile)
    finally:
        path.unlink(missing_ok=True)


def _load_llama3() -> dict:
    return load_recipe_yaml(LLAMA3_EXAMPLE, validate_model_layout=False)


def test_llama3_template_precision_fp8_block() -> None:
    raw = apply_recipe_cli_overrides(_load_llama3(), precision="fp8_block")
    plan = _compile_doc(raw, "nvidia_h20")
    assert plan.intent.abstract_scheme == "fp8_block"


def test_llama3_template_calib_preset_fast() -> None:
    raw = apply_recipe_cli_overrides(_load_llama3(), calib_preset="fast")
    assert raw["calib"]["max_samples"] == 32
    plan = _compile_doc(raw, "nvidia_h20")
    assert plan.intent.backend_options.get("pipeline") is False


def test_llama3_template_ignore_override() -> None:
    raw = apply_recipe_cli_overrides(
        _load_llama3(),
        ignore=["lm_head", "custom.*"],
    )
    plan = _compile_doc(raw, "nvidia_h20")
    assert "custom.*" in plan.intent.ignore


def test_llama3_precision_w4a16_overlay() -> None:
    raw = apply_recipe_cli_overrides(_load_llama3(), precision="w4a16")
    plan = _compile_doc(raw, "nvidia_h20")
    assert plan.intent.abstract_scheme == "w4a16"
    assert plan.intent.backend_options.get("algorithm") == "gptq"
    assert plan.intent.calib.get("max_samples") == 64


def test_llama3_pipeline_preset_fp8_then_gptq() -> None:
    from luban_sculpt.pipeline.config import parse_pipeline_config

    raw = apply_recipe_cli_overrides(
        _load_llama3(),
        pipeline_preset="fp8_then_gptq",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
    )
    spec = parse_pipeline_config(raw)
    assert len(spec.enabled_stages()) == 2
    assert spec.stages[1].backend == "gptq"
