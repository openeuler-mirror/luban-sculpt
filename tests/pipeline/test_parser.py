"""Unit tests for pipeline/config.py (parse_pipeline_config, build_stage_recipe)."""

from __future__ import annotations

import pytest
import yaml

from luban_sculpt.pipeline.config import (
    QuantStageConfig,
    build_stage_recipe,
    parse_pipeline_config,
)
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


def test_parse_missing_pipeline_raises() -> None:
    with pytest.raises(ValueError, match="requires pipeline.stages"):
        parse_pipeline_config({"model_id": "m"})


def test_parse_top_level_quant_raises() -> None:
    with pytest.raises(ValueError, match="quant.*removed"):
        parse_pipeline_config({"quant": {"backend": "gptq"}})


def test_parse_pipeline_list_shorthand_raises() -> None:
    with pytest.raises(ValueError, match="pipeline.stages"):
        parse_pipeline_config(
            {
                "pipeline": [
                    {"backend": "awq", "abstract_scheme": "w4a16_awq"},
                ]
            }
        )


def test_parse_pipeline_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="stages"):
        parse_pipeline_config({"pipeline": {}})


def test_parse_pipeline_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="mapping with 'stages'"):
        parse_pipeline_config({"pipeline": "sequential"})


def test_parse_stage_backend_block_merged_into_options() -> None:
    spec = parse_pipeline_config(
        {
            "pipeline": {
                "stages": [
                    {
                        "backend": "gptq",
                        "backend_options": {"group_size": 64},
                        "gptq": {"bits": 4},
                    }
                ]
            }
        }
    )
    st = spec.stages[0]
    assert st.backend_options["bits"] == 4
    assert st.backend_options["group_size"] == 64


def test_parse_stage_merges_backend_cfg_over_backend_options() -> None:
    spec = parse_pipeline_config(
        {
            "pipeline": {
                "stages": [
                    {
                        "name": "default",
                        "backend": "gptq",
                        "abstract_scheme": "w4_gptq",
                        "backend_options": {"bits": 8, "group_size": 64},
                        "gptq": {"bits": 4, "sym": True},
                    }
                ]
            }
        }
    )
    st = spec.stages[0]
    assert st.name == "default"
    assert st.backend == "gptq"
    assert st.backend_options["bits"] == 4
    assert st.backend_options["group_size"] == 64
    assert st.backend_options["sym"] is True


def test_parse_single_stage_default_backend_and_algorithm() -> None:
    spec = parse_pipeline_config(
        {
            "pipeline": {
                "stages": [
                    {
                        "backend": "llm_compressor",
                        "abstract_scheme": "fp8_dynamic",
                        "algorithm": "smoothquant",
                    }
                ]
            }
        }
    )
    st = spec.stages[0]
    assert st.backend == "llm_compressor"
    assert st.algorithm == "smoothquant"
    assert st.abstract_scheme == "fp8_dynamic"
    assert st.input_from == "recipe"


def test_parse_stage_empty_ignore_becomes_none() -> None:
    spec = parse_pipeline_config(
        {
            "pipeline": {
                "stages": [{"backend": "gptq", "ignore": []}],
            }
        }
    )
    assert spec.stages[0].ignore is None


def test_parse_stage_algo_field_becomes_algorithm() -> None:
    spec = parse_pipeline_config(
        {"pipeline": {"stages": [{"backend": "gptq", "algo": "gptq"}]}}
    )
    assert spec.stages[0].algorithm == "gptq"


def test_parse_stage_missing_backend_defaults_auto() -> None:
    spec = parse_pipeline_config({"pipeline": {"stages": [{"name": "x"}]}})
    assert spec.stages[0].backend == "auto"
    assert spec.stages[0].precision is None


def test_parse_stage_compress_overlay_decoupled_from_backend() -> None:
    spec = parse_pipeline_config(
        {
            "pipeline": {
                "stages": [
                    {
                        "backend": "auto",
                        "compress": {
                            "observer": {"weights": "minmax"},
                            "modifiers": [{"name": "QuantizationPatch", "mode": "patch"}],
                        },
                    }
                ]
            }
        }
    )
    st = spec.stages[0]
    assert st.compress_overlay["observer"] == {"weights": "minmax"}
    assert st.compress_overlay["modifiers"][0]["name"] == "QuantizationPatch"


def test_build_stage_recipe_passes_compress_overlay() -> None:
    stage = QuantStageConfig(
        name="s",
        backend="auto",
        compress_overlay={"observer": {"input": "minmax"}},
    )
    out = build_stage_recipe({"model_id": "m"}, stage, model_id="m")
    assert out["quant"]["compress_overlay"]["observer"] == {"input": "minmax"}


def test_parse_stage_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        parse_pipeline_config({"pipeline": {"stages": ["not-a-dict"]}})


def test_build_stage_recipe_writes_quant_view() -> None:
    stage = QuantStageConfig(
        name="s",
        backend="gptq",
        abstract_scheme="w4_gptq",
        backend_options={"bits": 4, "group_size": -1},
    )
    out = build_stage_recipe(
        {"model_id": "hub/model", "calib": {"max_samples": 1}},
        stage,
        model_id="hub/model",
    )
    assert out["quant"]["gptq"]["bits"] == 4
    assert out["quant"]["gptq"]["group_size"] == -1


def test_build_stage_recipe_strips_pipeline_and_sets_model_id() -> None:
    base = {
        "model_id": "orig",
        "pipeline": {"stages": []},
        "calib": {"max_samples": 1},
    }
    stage = QuantStageConfig(name="s", backend="llm_compressor")
    out = build_stage_recipe(base, stage, model_id="/data/stage0")
    assert "pipeline" not in out
    assert out["model_id"] == "/data/stage0"
    assert out["calib"]["max_samples"] == 1


def test_parse_packaged_multi_stage_recipe() -> None:
    from luban_sculpt.pipeline.recipe_overrides import apply_recipe_cli_overrides

    from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml
    from tests.paths import LLAMA3_EXAMPLE

    recipe = apply_recipe_cli_overrides(
        load_recipe_yaml(LLAMA3_EXAMPLE, validate_model_layout=False),
        pipeline_preset="fp8_then_gptq",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
    )
    spec = parse_pipeline_config(recipe)
    assert len(spec.enabled_stages()) == 2
    assert spec.stages[0].name == "fp8_prep"
    assert spec.stages[0].output_subdir == "stage1_fp8"
    assert spec.stages[1].name == "gptq_w4"
    assert spec.stages[1].algorithm == "gptq"
    assert spec.stages[1].input_from == "previous"
