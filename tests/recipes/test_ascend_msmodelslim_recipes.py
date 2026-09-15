"""Packaged Qwen + msModelSlim generic recipes: parse, compile, CLI argv."""

from __future__ import annotations

import pytest

from luban_sculpt.backends.msmodelslim.runner import build_quant_argv, resolve_quant_type
from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml
from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent
from luban_sculpt.pipeline.config import parse_pipeline_config
from luban_sculpt.pipeline.recipe_overrides import apply_recipe_cli_overrides
from tests.paths import QWEN25_EXAMPLE
from tests.recipe_materialize import materialize_recipe

ASCEND_PROFILE = "ascend_910b"

ASCEND_MSMODELSLIM_RECIPES = pytest.param(
    "w8a8",
    "w8a8",
    "ascend_w8a8",
    "w8a8",
    id="w8a8",
)
ASCEND_FP8_DYNAMIC_ALIAS = pytest.param(
    "fp8_dynamic",
    "fp8_dynamic",
    "ascend_w8a8",
    "w8a8",
    id="fp8_dynamic_alias_w8a8",
)
ASCEND_FP8_BLOCK_ALIAS = pytest.param(
    "fp8_block",
    "fp8_block",
    "ascend_w4a8",
    "w4a8",
    id="fp8_block_alias_w4a8",
)


@pytest.mark.parametrize(
    ("precision", "expected_precision", "expected_scheme", "expected_quant_type"),
    [
        ASCEND_MSMODELSLIM_RECIPES,
        ASCEND_FP8_DYNAMIC_ALIAS,
        ASCEND_FP8_BLOCK_ALIAS,
    ],
)
def test_parse_ascend_msmodelslim_pipeline_recipe(
    precision: str,
    expected_precision: str,
    expected_scheme: str,
    expected_quant_type: str,
) -> None:
    del expected_scheme, expected_quant_type
    doc = load_recipe_yaml(QWEN25_EXAMPLE, validate_model_layout=False)
    doc = apply_recipe_cli_overrides(doc, precision=precision)
    assert "quant" not in doc

    spec = parse_pipeline_config(doc)
    assert len(spec.enabled_stages()) == 1
    stage = spec.stages[0]
    assert stage.backend == "auto"
    assert stage.precision == expected_precision


@pytest.mark.parametrize(
    ("precision", "expected_precision", "expected_scheme", "expected_quant_type"),
    [
        ASCEND_MSMODELSLIM_RECIPES,
        ASCEND_FP8_DYNAMIC_ALIAS,
        ASCEND_FP8_BLOCK_ALIAS,
    ],
)
def test_compile_ascend_msmodelslim_recipe(
    tmp_path,
    precision: str,
    expected_precision: str,
    expected_scheme: str,
    expected_quant_type: str,
) -> None:
    del expected_precision
    recipe = materialize_recipe(
        QWEN25_EXAMPLE,
        tmp_path / f"qwen-{precision}.yaml",
        precision=precision,
    )
    plan = compile_plan(recipe, ASCEND_PROFILE)
    assert plan.intent.backend == "msmodelslim"
    assert plan.intent.abstract_scheme == expected_scheme
    assert plan.intent.model_arch.value == "qwen"
    assert plan.export_format == ExportFormat.VLLM_ASCEND
    assert plan.intent.backend_options.get("quant_type") == expected_quant_type


def test_ascend_fp8_block_merges_qwen_gate_ignore(tmp_path) -> None:
    recipe = materialize_recipe(
        QWEN25_EXAMPLE,
        tmp_path / "qwen-fp8-block.yaml",
        precision="fp8_block",
    )
    plan = compile_plan(recipe, ASCEND_PROFILE)
    ignore = plan.intent.ignore
    assert "lm_head" in ignore
    assert "re:.*mlp.gate$" in ignore


def test_resolve_quant_type_from_abstract_scheme_without_explicit_quant_type(
    tmp_path,
) -> None:
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
    argv = build_quant_argv(plan, tmp_path / "out")
    assert "--quant_type" in argv
    assert argv[argv.index("--quant_type") + 1] == "w4a8"
