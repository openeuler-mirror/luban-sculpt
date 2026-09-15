"""CPU profile（generic_cpu）dry-run 测试用例。

本机无卡也可跑::

    pytest example/cpu/test_cpu_dry_run.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.contracts import ExportFormat
from luban_sculpt.hae.engine import HardwareAwareEngine
from luban_sculpt.pipeline import QuantPipeline

_ROOT = Path(__file__).resolve().parents[2]
_PKG = _ROOT / "luban_sculpt"
LLAMA3_EXAMPLE = _PKG / "recipes" / "llama3.yaml"
PROFILE = "generic_cpu"


@pytest.fixture(autouse=True)
def _dry_run_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUBAN_LLM_COMPRESSOR_DRY_RUN", "1")
    monkeypatch.delenv("LUBAN_PROBE_FAIL_OPS", raising=False)


def test_cpu_profile_probe() -> None:
    hw, probe, profile = HardwareAwareEngine(PROFILE).run(PROFILE)
    assert hw.profile_id == PROFILE
    assert probe.ok is True
    assert probe.missing_ops == []
    assert "fp8_dynamic" in profile.get("schemes", {})
    assert "w4_gptq" in profile.get("schemes", {})


def test_cpu_compile_llama_fp8_recipe() -> None:
    plan = compile_plan(LLAMA3_EXAMPLE, PROFILE)
    assert plan.hw.profile_id == PROFILE
    assert plan.intent.backend == "llm_compressor"
    assert plan.intent.abstract_scheme == "fp8_dynamic"
    assert plan.export_format == ExportFormat.COMPRESSED_TENSORS


def test_cpu_compress_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "cpu_out"
    artifact = QuantPipeline(profile_name=PROFILE).run(
        LLAMA3_EXAMPLE, out
    )
    manifest_path = artifact.output_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile_id"] == PROFILE
    assert manifest["backend"] == "llm_compressor"
    assert manifest["export_format"] == "compressed-tensors"
    assert manifest["abstract_scheme"] == "fp8_dynamic"


def test_cpu_pipeline_fp8_then_gptq_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LUBAN_GPTQMODEL_DRY_RUN", "1")
    from luban_sculpt.pipeline.recipe_overrides import apply_recipe_cli_overrides

    from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml

    recipe = apply_recipe_cli_overrides(
        load_recipe_yaml(LLAMA3_EXAMPLE, validate_model_layout=False),
        pipeline_preset="fp8_then_gptq",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
    )
    out = tmp_path / "pipe"
    artifact = QuantPipeline(profile_name=PROFILE).run_recipe(recipe, out)
    assert (out / "pipeline_manifest.json").is_file()
    assert (out / "stage1_fp8" / "manifest.json").is_file()
    assert (out / "stage2_gptq" / "manifest.json").is_file()
    assert artifact.output_dir == out / "stage2_gptq"
