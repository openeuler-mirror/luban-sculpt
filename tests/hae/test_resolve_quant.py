"""HAE resolve_quant: precision → scheme → backend."""

from __future__ import annotations

import pytest

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.contracts import ExportFormat
from luban_sculpt.hae.engine import load_profile_template
from luban_sculpt.hae.resolve_quant import (
    resolve_compress_backend,
    resolve_scheme_for_profile,
    suggest_compress_route,
)
from tests.paths import QWEN25_EXAMPLE
from tests.recipe_materialize import materialize_recipe


@pytest.mark.parametrize(
    "profile_name,precision,expected_scheme,expected_backend",
    [
        ("nvidia_h20", "fp8_dynamic", "fp8_dynamic", "llm_compressor"),
        ("ascend_910b", "fp8_dynamic", "ascend_w8a8", "msmodelslim"),
        ("ascend_910b", "fp8_block", "ascend_w4a8", "msmodelslim"),
        ("nvidia_h20", "w4a16", "w4a16", "llm_compressor"),
    ],
)
def test_suggest_compress_route(
    profile_name: str,
    precision: str,
    expected_scheme: str,
    expected_backend: str,
) -> None:
    profile = load_profile_template(profile_name)
    route = suggest_compress_route(profile, precision=precision)
    assert route.abstract_scheme == expected_scheme
    assert route.backend == expected_backend


def test_generic_recipe_auto_backend_h20(tmp_path) -> None:
    recipe = materialize_recipe(QWEN25_EXAMPLE, tmp_path / "qwen.yaml")
    plan = compile_plan(recipe, "nvidia_h20")
    assert plan.intent.backend == "llm_compressor"
    assert plan.intent.abstract_scheme == "fp8_dynamic"
    assert plan.export_format == ExportFormat.COMPRESSED_TENSORS
    opts = plan.intent.backend_options
    assert opts.get("trust_remote_code") is True
    assert opts.get("device_map") == "cuda:0"
    assert opts.get("observer") == {
        "weights": "luban_ema_absmax",
        "input": "minmax",
    }
    assert opts["modifiers"][0]["name"] == "QuantizationPatch"


def test_generic_recipe_auto_backend_ascend(tmp_path) -> None:
    recipe = materialize_recipe(QWEN25_EXAMPLE, tmp_path / "qwen.yaml")
    plan = compile_plan(recipe, "ascend_910b")
    assert plan.intent.backend == "msmodelslim"
    assert plan.intent.abstract_scheme == "ascend_w8a8"
    opts = plan.intent.backend_options
    assert opts.get("quant_type") == "w8a8"
    assert opts.get("device") == "npu"
    assert opts.get("model_type") == "Qwen2.5-7B-Instruct"
    assert plan.export_format == ExportFormat.VLLM_ASCEND


def test_backend_override_wins() -> None:
    profile = load_profile_template("nvidia_h20")
    backend = resolve_compress_backend(
        profile, "fp8_dynamic", backend_hint="gptq"
    )
    assert backend == "gptq"


def test_unknown_precision_raises() -> None:
    profile = load_profile_template("ascend_910b")
    with pytest.raises(ValueError, match="cannot map"):
        resolve_scheme_for_profile(profile, "not_a_real_precision_xyz")
