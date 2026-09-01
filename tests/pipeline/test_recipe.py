"""Unit tests for pipeline/recipe.py (parse_pipeline_config, build_stage_recipe)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from luban_sculpt.pipeline.recipe import build_stage_recipe, parse_pipeline_config
from luban_sculpt.pipeline.config import QuantStageConfig
from tests.paths import RECIPES


def test_parse_pipeline_stages_dict() -> None:
    spec = parse_pipeline_config(
        {
            "pipeline": {
                "stages": [
                    {"name": "s0", "backend": "llm_compressor"},
                    {"name": "s1", "backend": "gptq", "input_from": "previous"},
                ]
            }
        }
    )
    assert len(spec.stages) == 2
    assert spec.stages[0].name == "s0"
    assert spec.stages[1].input_from == "previous"


def test_parse_pipeline_list_shorthand() -> None:
    spec = parse_pipeline_config(
        {
            "pipeline": [
                {"backend": "awq", "abstract_scheme": "w4a16_awq"},
                {"backend": "gptq", "name": "second"},
            ]
        }
    )
    assert spec.stages[0].name == "stage_0"
    assert spec.stages[0].backend == "awq"
    assert spec.stages[1].name == "second"


def test_parse_single_quant_merges_backend_cfg_into_options() -> None:
    """quant.gptq（backend_cfg）与 backend_options 合并，专属块覆盖同名键。"""
    spec = parse_pipeline_config(
        {
            "quant": {
                "backend": "gptq",
                "abstract_scheme": "w4_gptq",
                "backend_options": {"bits": 8, "group_size": 64},
                "gptq": {"bits": 4, "sym": True},
            }
        }
    )
    st = spec.stages[0]
    assert st.name == "default"
    assert st.backend == "gptq"
    assert st.backend_options["bits"] == 4
    assert st.backend_options["group_size"] == 64
    assert st.backend_options["sym"] is True


def test_parse_single_quant_default_backend_and_algorithm() -> None:
    spec = parse_pipeline_config(
        {
            "quant": {
                "abstract_scheme": "fp8_dynamic",
                "algorithm": "smoothquant",
            }
        }
    )
    st = spec.stages[0]
    assert st.backend == "llm_compressor"
    assert st.algorithm == "smoothquant"
    assert st.abstract_scheme == "fp8_dynamic"
    assert st.input_from == "recipe"


def test_parse_single_quant_empty_ignore_becomes_none() -> None:
    spec = parse_pipeline_config({"quant": {"backend": "gptq", "ignore": []}})
    assert spec.stages[0].ignore is None


def test_parse_stage_algo_field_becomes_algorithm() -> None:
    spec = parse_pipeline_config(
        {"pipeline": {"stages": [{"backend": "gptq", "algo": "gptq"}]}}
    )
    assert spec.stages[0].algorithm == "gptq"


def test_parse_stage_missing_backend_raises() -> None:
    with pytest.raises(ValueError, match="missing required field 'backend'"):
        parse_pipeline_config({"pipeline": {"stages": [{"name": "x"}]}})


def test_parse_stage_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        parse_pipeline_config({"pipeline": {"stages": ["not-a-dict"]}})


def test_build_stage_recipe_merges_base_backend_cfg() -> None:
    base = {
        "model_id": "hub/model",
        "quant": {
            "backend": "gptq",
            "gptq": {"bits": 8, "group_size": -1},
        },
    }
    stage = QuantStageConfig(
        name="s",
        backend="gptq",
        abstract_scheme="w4_gptq",
        backend_options={"bits": 4},
    )
    out = build_stage_recipe(base, stage, model_id="hub/model")
    assert out["quant"]["gptq"]["bits"] == 4
    assert out["quant"]["gptq"]["group_size"] == -1


def test_build_stage_recipe_strips_pipeline_and_sets_model_id() -> None:
    base = {
        "model_id": "orig",
        "pipeline": {"stages": []},
        "quant": {"backend": "llm_compressor"},
        "calib": {"max_samples": 1},
    }
    stage = QuantStageConfig(name="s", backend="llm_compressor")
    out = build_stage_recipe(base, stage, model_id="/data/stage0")
    assert "pipeline" not in out
    assert out["model_id"] == "/data/stage0"
    assert out["calib"]["max_samples"] == 1


def test_parse_packaged_gptq_recipe() -> None:
    with (RECIPES / "h20_qwen_gptq_w4.yaml").open(encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    spec = parse_pipeline_config(recipe)
    assert len(spec.stages) == 1
    st = spec.stages[0]
    assert st.backend == "gptq"
    assert st.abstract_scheme == "w4_gptq"
    assert st.backend_options.get("bits") == 4
    assert st.backend_options.get("group_size") == 128


def test_parse_packaged_multi_stage_recipe() -> None:
    with (RECIPES / "pipeline_llm_compressor_then_gptq.yaml").open(encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    spec = parse_pipeline_config(recipe)
    assert len(spec.enabled_stages()) == 2
    assert spec.stages[0].name == "fp8_prep"
    assert spec.stages[0].output_subdir == "stage1_fp8"
    assert spec.stages[1].name == "gptq_w4"
    assert spec.stages[1].algorithm == "gptq"
    assert spec.stages[1].input_from == "previous"
