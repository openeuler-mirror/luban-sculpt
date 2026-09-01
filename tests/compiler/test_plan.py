"""Tests for recipe YAML + profile → BackendPlan compilation."""

from __future__ import annotations

from pathlib import Path

import pytest

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.contracts import BackendPlan, ExportFormat
from tests.paths import RECIPES


@pytest.mark.parametrize(
    "recipe_file,profile_name,expected_backend,expected_scheme,expected_export",
    [
        (
            "h20_qwen_fp8_dynamic.yaml",
            "nvidia_h20",
            "llm_compressor",
            "fp8_dynamic",
            ExportFormat.COMPRESSED_TENSORS,
        ),
        (
            "h20_qwen_gptq_w4.yaml",
            "nvidia_h20",
            "gptq",
            "w4_gptq",
            ExportFormat.GPTQ_HF,
        ),
        (
            "ascend_qwen_w8a8.yaml",
            "ascend_910b",
            "msmodelslim",
            "ascend_w8a8",
            ExportFormat.VLLM_ASCEND,
        ),
        (
            "hygon_qwen_w4a16_awq.yaml",
            "hygon_dcu",
            "awq",
            "hygon_w4a16_awq",
            ExportFormat.AWQ_HF,
        ),
        (
            "hygon_qwen_w8a8_gptq.yaml",
            "hygon_dcu",
            "gptq",
            "hygon_w8a8_gptq",
            ExportFormat.GPTQ_HF,
        ),
        (
            "llama_fp8_dynamic.yaml",
            "generic_cpu",
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


def test_backend_options_from_named_quant_block() -> None:
    plan = compile_plan(RECIPES / "h20_qwen_fp8_dynamic.yaml", "nvidia_h20")

    assert plan.intent.backend_options.get("scheme") == "FP8_DYNAMIC"
    assert plan.intent.backend_options.get("trust_remote_code") is True
    modifiers = plan.intent.backend_options.get("modifiers") or []
    assert any(m.get("name") == "HALCalibHook" for m in modifiers)


def test_gptq_backend_options_bits_and_group_size() -> None:
    plan = compile_plan(RECIPES / "h20_qwen_gptq_w4.yaml", "nvidia_h20")

    opts = plan.intent.backend_options
    assert opts.get("bits") == 4
    assert opts.get("group_size") == 128
    assert plan.intent.deploy_target == "vllm_cuda"


def test_profile_wire_overrides_backend_default() -> None:
    """ascend_w8a8 profile export=vllm_ascend, not llm_compressor CT."""
    plan = compile_plan(RECIPES / "ascend_qwen_w8a8.yaml", "ascend_910b")

    assert plan.intent.backend_options.get("quant_type") == "w8a8"
    assert plan.export_format == ExportFormat.VLLM_ASCEND


def test_hygon_awq_backend_options() -> None:
    plan = compile_plan(RECIPES / "hygon_qwen_w4a16_awq.yaml", "hygon_dcu")

    opts = plan.intent.backend_options
    assert opts.get("trust_remote_code") is True
    assert plan.intent.deploy_target == "vllm_rocm"
    assert plan.export_format == ExportFormat.AWQ_HF


def test_hygon_gptq_export_format() -> None:
    plan = compile_plan(RECIPES / "hygon_qwen_w8a8_gptq.yaml", "hygon_dcu")

    assert plan.intent.backend_options.get("bits") == 8
    assert plan.export_format == ExportFormat.GPTQ_HF


def test_compiled_plans_expose_model_arch() -> None:
    plan = compile_plan(RECIPES / "llama_fp8_dynamic.yaml", "generic_cpu")
    assert plan.intent.model_arch.value == "llama"

    qwen = compile_plan(RECIPES / "ascend_qwen_w8a8.yaml", "ascend_910b")
    assert qwen.intent.model_arch.value == "qwen"


def test_all_packaged_recipes_compile_with_matching_profile() -> None:
    """Smoke: every recipe under recipes/ compiles with at least one profile."""
    mapping = {
        "h20_qwen_fp8_dynamic.yaml": "nvidia_h20",
        "h20_qwen_fp8_block.yaml": "nvidia_h20",
        "h20_qwen_gptq_w4.yaml": "nvidia_h20",
        "llama_fp8_dynamic.yaml": "generic_cpu",
        "moe_int4.yaml": "generic_cpu",
        "ascend_qwen_w8a8.yaml": "ascend_910b",
        "hygon_qwen_w4a16_awq.yaml": "hygon_dcu",
        "hygon_qwen_w8a8_gptq.yaml": "hygon_dcu",
    }
    for name, profile_name in mapping.items():
        path = RECIPES / name
        if not path.is_file():
            pytest.skip(f"missing {name}")
        plan = compile_plan(path, profile_name)
        assert plan.intent.model_id
        assert plan.export_format is not None
