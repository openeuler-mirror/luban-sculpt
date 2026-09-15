"""Tests for recipe YAML + profile → BackendPlan compilation."""

from __future__ import annotations

import pytest

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.contracts import BackendPlan, ExportFormat
from tests.paths import LLAMA3_EXAMPLE, QWEN25_EXAMPLE, RECIPES
from tests.recipe_materialize import materialize_recipe


@pytest.mark.parametrize(
    "recipe_path,profile_name,precision,expected_backend,expected_scheme,expected_export",
    [
        (
            QWEN25_EXAMPLE,
            "ascend_910b",
            "w8a8",
            "msmodelslim",
            "ascend_w8a8",
            ExportFormat.VLLM_ASCEND,
        ),
        (
            LLAMA3_EXAMPLE,
            "nvidia_h20",
            None,
            "llm_compressor",
            "fp8_dynamic",
            ExportFormat.COMPRESSED_TENSORS,
        ),
    ],
)
def test_compile_recipe_to_plan(
    tmp_path,
    recipe_path,
    profile_name: str,
    precision: str | None,
    expected_backend: str,
    expected_scheme: str,
    expected_export: ExportFormat,
) -> None:
    recipe = materialize_recipe(
        recipe_path,
        tmp_path / "recipe.yaml",
        precision=precision,
    )
    plan = compile_plan(recipe, profile_name)

    assert isinstance(plan, BackendPlan)
    assert plan.intent.backend == expected_backend
    assert plan.intent.abstract_scheme == expected_scheme
    assert plan.export_format == expected_export
    assert plan.hw.profile_id == profile_name


def test_profile_wire_overrides_backend_default(tmp_path) -> None:
    """ascend_w8a8 profile export=vllm_ascend, not llm_compressor CT."""
    recipe = materialize_recipe(
        QWEN25_EXAMPLE,
        tmp_path / "qwen-w8a8.yaml",
        precision="w8a8",
    )
    plan = compile_plan(recipe, "ascend_910b")

    assert plan.intent.backend_options.get("quant_type") == "w8a8"
    assert plan.export_format == ExportFormat.VLLM_ASCEND


def test_compiled_plans_expose_model_arch(tmp_path) -> None:
    llama = materialize_recipe(LLAMA3_EXAMPLE, tmp_path / "llama.yaml")
    plan = compile_plan(llama, "nvidia_h20")
    assert plan.intent.model_arch.value == "llama"

    qwen = materialize_recipe(
        QWEN25_EXAMPLE,
        tmp_path / "qwen.yaml",
        precision="w8a8",
    )
    qwen_plan = compile_plan(qwen, "ascend_910b")
    assert qwen_plan.intent.model_arch.value == "qwen"


@pytest.mark.parametrize(
    "recipe_path,profile_name,precision",
    [
        (QWEN25_EXAMPLE, "nvidia_h20", "fp8_dynamic"),
        (QWEN25_EXAMPLE, "ascend_910b", "fp8_dynamic"),
        (QWEN25_EXAMPLE, "ascend_910b", "w8a8"),
        (QWEN25_EXAMPLE, "ascend_910b", "fp8_block"),
        (QWEN25_EXAMPLE, "nvidia_h20", "fp8_block"),
        (LLAMA3_EXAMPLE, "nvidia_h20", None),
        (RECIPES / "qwen_observer_smoke.yaml", "nvidia_h20", None),
        (LLAMA3_EXAMPLE, "generic_cpu", None),
        (RECIPES / "llama3_fp8_then_gptq.yaml", "generic_cpu", None),
    ],
)
def test_all_packaged_recipes_compile_with_matching_profile(
    tmp_path,
    recipe_path,
    profile_name: str,
    precision: str | None,
) -> None:
    if not recipe_path.is_file():
        pytest.skip(f"missing {recipe_path}")
    recipe = materialize_recipe(
        recipe_path,
        tmp_path / "recipe.yaml",
        precision=precision,
    )
    plan = compile_plan(recipe, profile_name)
    assert plan.intent.model_id
    assert plan.export_format is not None
