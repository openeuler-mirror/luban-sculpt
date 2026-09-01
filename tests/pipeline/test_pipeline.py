"""Pipeline composition / multi-backend orchestration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luban_sculpt.pipeline import (
    QuantPipeline,
    build_stage_recipe,
    parse_pipeline_config,
)
from luban_sculpt.pipeline.config import QuantStageConfig
from tests.paths import RECIPES


def test_parse_single_quant_compat() -> None:
    recipe = {
        "model_id": "m",
        "quant": {
            "backend": "gptq",
            "abstract_scheme": "hygon_w8a8_gptq",
            "gptq": {"bits": 8, "group_size": -1},
        },
    }
    spec = parse_pipeline_config(recipe)
    assert len(spec.stages) == 1
    st = spec.stages[0]
    assert st.backend == "gptq"
    assert st.abstract_scheme == "hygon_w8a8_gptq"
    assert st.backend_options.get("bits") == 8


def test_parse_multi_stage_pipeline() -> None:
    recipe = {
        "model_id": "m",
        "pipeline": {
            "stages": [
                {
                    "name": "a",
                    "backend": "llm_compressor",
                    "abstract_scheme": "fp8_dynamic",
                },
                {
                    "name": "b",
                    "backend": "gptq",
                    "algo": "gptq",
                    "abstract_scheme": "w4_gptq",
                    "input_from": "previous",
                },
            ]
        },
    }
    spec = parse_pipeline_config(recipe)
    assert [s.backend for s in spec.stages] == ["llm_compressor", "gptq"]
    assert spec.stages[1].algorithm == "gptq"
    assert spec.stages[1].input_from == "previous"


def test_build_stage_recipe_injects_algorithm() -> None:
    base = {
        "model_id": "orig",
        "quant": {"backend": "gptq"},
        "calib": {"max_samples": 8},
    }
    stage = QuantStageConfig(
        name="s",
        backend="gptq",
        algorithm="gptq",
        abstract_scheme="w4_gptq",
        backend_options={"bits": 4},
    )
    out = build_stage_recipe(base, stage, model_id="/tmp/prev")
    assert out["model_id"] == "/tmp/prev"
    assert out["quant"]["backend"] == "gptq"
    assert out["quant"]["gptq"]["algo"] == "gptq"
    assert out["quant"]["gptq"]["bits"] == 4
    assert out["calib"]["max_samples"] == 8


def test_pipeline_dry_run_multi_stage(tmp_path: Path) -> None:
    import os

    os.environ["LUBAN_LLM_COMPRESSOR_DRY_RUN"] = "1"
    os.environ["LUBAN_GPTQMODEL_DRY_RUN"] = "1"
    recipe = RECIPES / "pipeline_llm_compressor_then_gptq.yaml"
    out = tmp_path / "out"
    pipe = QuantPipeline(profile_name="generic_cpu")
    artifact = pipe.run(recipe, out)
    assert (out / "pipeline_manifest.json").is_file()
    assert (out / "stage1_fp8" / "manifest.json").is_file()
    assert (out / "stage2_gptq" / "manifest.json").is_file()
    assert artifact.output_dir == out / "stage2_gptq"


def test_single_recipe_still_works_via_pipeline(tmp_path: Path) -> None:
    import os

    os.environ["LUBAN_LLM_COMPRESSOR_DRY_RUN"] = "1"
    recipe = RECIPES / "llama_fp8_dynamic.yaml"
    out = tmp_path / "single"
    artifact = QuantPipeline(profile_name="generic_cpu").run(recipe, out)
    assert artifact.output_dir.is_dir()
    assert (artifact.output_dir / "manifest.json").is_file()


def test_pipeline_with_artifact_validate(tmp_path: Path) -> None:
    import os

    os.environ["LUBAN_LLM_COMPRESSOR_DRY_RUN"] = "1"
    recipe = RECIPES / "llama_fp8_dynamic.yaml"
    out = tmp_path / "validated"
    artifact = QuantPipeline(
        profile_name="generic_cpu",
        validate_quantized_model=True,
    ).run(recipe, out)
    assert (artifact.output_dir / "manifest.json").is_file()
    report = artifact.output_dir / "quantized_model_validate.json"
    assert report.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["check_hf_config"] is True
