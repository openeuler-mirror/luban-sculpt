"""Packaged two-stage recipe (llama3_fp8_then_gptq.yaml)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml
from luban_sculpt.pipeline import QuantPipeline
from luban_sculpt.pipeline.config import parse_pipeline_config
from tests.paths import LLAMA3_FP8_THEN_GPTQ, RECIPES


def test_two_stage_recipe_yaml_parses() -> None:
    doc = load_recipe_yaml(LLAMA3_FP8_THEN_GPTQ, validate_model_layout=False)
    spec = parse_pipeline_config(doc)
    stages = spec.enabled_stages()
    assert len(stages) == 2
    assert stages[0].name == "fp8_prep"
    assert stages[0].precision == "fp8_dynamic"
    assert stages[0].output_subdir == "stage1_fp8"
    assert stages[1].name == "gptq_w4"
    assert stages[1].backend == "gptq"
    assert stages[1].input_from == "previous"
    assert stages[1].output_subdir == "stage2_gptq"


def test_two_stage_recipe_first_stage_compiles() -> None:
    plan = compile_plan(LLAMA3_FP8_THEN_GPTQ, "generic_cpu")
    assert plan.intent.backend == "llm_compressor"
    assert plan.intent.abstract_scheme == "fp8_dynamic"
    assert plan.intent.model_arch.value == "llama"


@pytest.fixture(autouse=True)
def _dry_run_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUBAN_LLM_COMPRESSOR_DRY_RUN", "1")
    monkeypatch.setenv("LUBAN_GPTQMODEL_DRY_RUN", "1")


def test_two_stage_recipe_pipeline_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "two_stage"
    artifact = QuantPipeline(
        profile_name="generic_cpu",
        validate_quantized_model=True,
    ).run(LLAMA3_FP8_THEN_GPTQ, out)

    pipe_manifest = json.loads((out / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert len(pipe_manifest["stages"]) == 2
    assert pipe_manifest["stages"][0]["backend"] == "llm_compressor"
    assert pipe_manifest["stages"][1]["backend"] == "gptq"
    assert pipe_manifest["final_output"].endswith("stage2_gptq")

    assert (out / "stage1_fp8" / "manifest.json").is_file()
    assert (out / "stage2_gptq" / "manifest.json").is_file()
    assert artifact.output_dir == out / "stage2_gptq"
    validate = json.loads(
        (out / "stage2_gptq" / "quantized_model_validate.json").read_text(encoding="utf-8")
    )
    assert validate["ok"] is True


def test_two_stage_recipe_packaged_in_recipes_dir() -> None:
    assert (RECIPES / "llama3_fp8_then_gptq.yaml").is_file()
