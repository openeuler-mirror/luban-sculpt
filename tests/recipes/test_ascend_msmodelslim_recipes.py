"""Packaged Ascend Qwen + msModelSlim recipes: parse, compile, CLI argv."""

from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from luban_sculpt.backends.msmodelslim.runner import build_quant_argv, resolve_quant_type
from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent
from luban_sculpt.pipeline.config import parse_pipeline_config
from tests.paths import RECIPES

ASCEND_PROFILE = "ascend_910b"

ASCEND_MSMODELSLIM_RECIPES = pytest.param(
    "ascend_qwen_w8a8.yaml",
    "ascend_w8a8",
    "w8a8",
    id="w8a8",
)
ASCEND_FP8_DYNAMIC_ALIAS = pytest.param(
    "ascend_qwen_fp8_dynamic.yaml",
    "ascend_w8a8",
    "w8a8",
    id="fp8_dynamic_alias_w8a8",
)
ASCEND_FP8_BLOCK_ALIAS = pytest.param(
    "ascend_qwen_fp8_block.yaml",
    "ascend_w4a8",
    "w4a8",
    id="fp8_block_alias_w4a8",
)


@pytest.mark.parametrize(
    ("recipe_file", "expected_scheme", "expected_quant_type"),
    [
        ASCEND_MSMODELSLIM_RECIPES,
        ASCEND_FP8_DYNAMIC_ALIAS,
        ASCEND_FP8_BLOCK_ALIAS,
    ],
)
def test_parse_ascend_msmodelslim_pipeline_recipe(
    recipe_file: str,
    expected_scheme: str,
    expected_quant_type: str,
) -> None:
    path = RECIPES / recipe_file
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert "quant" not in doc

    spec = parse_pipeline_config(doc)
    assert len(spec.enabled_stages()) == 1
    stage = spec.stages[0]
    assert stage.backend == "msmodelslim"
    assert stage.abstract_scheme == expected_scheme
    assert stage.backend_options.get("quant_type") == expected_quant_type
    assert stage.backend_options.get("model_type") == "Qwen2.5-7B-Instruct"


@pytest.mark.parametrize(
    ("recipe_file", "expected_scheme", "expected_quant_type"),
    [
        ASCEND_MSMODELSLIM_RECIPES,
        ASCEND_FP8_DYNAMIC_ALIAS,
        ASCEND_FP8_BLOCK_ALIAS,
    ],
)
def test_compile_ascend_msmodelslim_recipe(
    recipe_file: str,
    expected_scheme: str,
    expected_quant_type: str,
) -> None:
    plan = compile_plan(RECIPES / recipe_file, ASCEND_PROFILE)
    assert plan.intent.backend == "msmodelslim"
    assert plan.intent.abstract_scheme == expected_scheme
    assert plan.intent.model_arch.value == "qwen"
    assert plan.export_format == ExportFormat.VLLM_ASCEND
    assert plan.intent.backend_options.get("quant_type") == expected_quant_type


def test_ascend_fp8_block_merges_qwen_gate_ignore() -> None:
    plan = compile_plan(RECIPES / "ascend_qwen_fp8_block.yaml", ASCEND_PROFILE)
    ignore = plan.intent.ignore
    assert "lm_head" in ignore
    assert "re:.*mlp.gate$" in ignore


def test_resolve_quant_type_from_abstract_scheme_without_explicit_quant_type() -> None:
    plan = BackendPlan(
        intent=QuantIntent(
            model_id="m",
            backend="msmodelslim",
            abstract_scheme="ascend_w4a8",
            infer_runtime="vllm_ascend",
            backend_options={"model_type": "Qwen2.5-7B-Instruct"},
        ),
        hw=HwDecision(profile_id=ASCEND_PROFILE),
        export_format=ExportFormat.VLLM_ASCEND,
    )
    assert resolve_quant_type(plan) == "w4a8"
    argv = build_quant_argv(plan, Path("/tmp/out"))
    assert "--quant_type" in argv
    assert argv[argv.index("--quant_type") + 1] == "w4a8"
