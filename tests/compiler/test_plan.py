"""Tests for recipe YAML + profile → BackendPlan compilation."""

from __future__ import annotations

import pytest

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.contracts import BackendPlan, ExportFormat
from tests.paths import RECIPES


@pytest.mark.parametrize(
    "recipe_file,profile_name,expected_backend,expected_scheme,expected_export",
    [
        (
            "ascend_qwen_w8a8.yaml",
            "ascend_910b",
            "msmodelslim",
            "ascend_w8a8",
            ExportFormat.VLLM_ASCEND,
        ),
        (
            "h20_llama3_fp8_dynamic.yaml",
            "nvidia_h20",
            "llm_compressor",
            "fp8_dynamic",
            ExportFormat.COMPRESSED_TENSORS,
        ),
    ],
)
def test_compile_recipe_to_plan(
    recipe_file: str,
    profile_name: str,
    expected_backend: str,
    expected_scheme: str,
    expected_export: ExportFormat,
) -> None:
    plan = compile_plan(RECIPES / recipe_file, profile_name)

    assert isinstance(plan, BackendPlan)
    assert plan.intent.backend == expected_backend
    assert plan.intent.abstract_scheme == expected_scheme
    assert plan.export_format == expected_export
    assert plan.hw.profile_id == profile_name


def test_profile_wire_overrides_backend_default() -> None:
    """ascend_w8a8 profile export=vllm_ascend, not llm_compressor CT."""
    plan = compile_plan(RECIPES / "ascend_qwen_w8a8.yaml", "ascend_910b")

    assert plan.intent.backend_options.get("quant_type") == "w8a8"
    assert plan.export_format == ExportFormat.VLLM_ASCEND


def test_compiled_plans_expose_model_arch() -> None:
    plan = compile_plan(RECIPES / "h20_llama3_fp8_dynamic.yaml", "nvidia_h20")
    assert plan.intent.model_arch.value == "llama"

    qwen = compile_plan(RECIPES / "ascend_qwen_w8a8.yaml", "ascend_910b")
    assert qwen.intent.model_arch.value == "qwen"


def test_all_packaged_recipes_compile_with_matching_profile() -> None:
    """Smoke: every recipe under recipes/ compiles with at least one profile."""
    mapping = {
        "h20_llama3_fp8_dynamic.yaml": "nvidia_h20",
        "h20_llama3_fp8_dynamic_fast.yaml": "nvidia_h20",
        "h20_llama3_fp8_block.yaml": "nvidia_h20",
        "h20_llama3_8b_fp8_dynamic.yaml": "nvidia_h20",
        "h20_llama3_w4a16.yaml": "nvidia_h20",
        "moe_int4.yaml": "generic_cpu",
        "ascend_qwen_w8a8.yaml": "ascend_910b",
        "ascend_qwen_fp8_dynamic.yaml": "ascend_910b",
        "ascend_qwen_fp8_block.yaml": "ascend_910b",
        "qwen_observer_smoke_test.yaml": "nvidia_h20",
        "pipeline_llm_compressor_then_gptq.yaml": "nvidia_h20",
    }
    for name, profile_name in mapping.items():
        path = RECIPES / name
        if not path.is_file():
            pytest.skip(f"missing {name}")
        plan = compile_plan(path, profile_name)
        assert plan.intent.model_id
        assert plan.export_format is not None
