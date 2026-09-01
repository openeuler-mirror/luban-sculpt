"""validate/ unit tests (hf_config + QuantizedModelValidateStage)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luban_sculpt.contracts import (
    ArtifactManifest,
    BackendPlan,
    ExportFormat,
    HwDecision,
    QuantIntent,
    QuantizedArtifact,
)
from luban_sculpt.pipeline.config import PipelineConfig, QuantStageConfig
from luban_sculpt.pipeline.context import PipelineContext, StageResult
from luban_sculpt.pipeline.stages import QuantizedModelValidateStage
from luban_sculpt.validate.hf_config import validate_hf_config
from luban_sculpt.validate.quantized_model import QuantizedModelValidateError


def test_validate_hf_config_gptq_without_manifest(tmp_path: Path) -> None:
    """标准社区 GPTQ 目录：有 config.json，无 luban manifest。"""
    model = tmp_path / "gptq_model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                }
            }
        ),
        encoding="utf-8",
    )
    assert validate_hf_config(model) == []


def test_quantized_model_validate_stage_missing_config_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    plan = BackendPlan(
        intent=QuantIntent(
            model_id="m",
            backend="llm_compressor",
            abstract_scheme="fp8_dynamic",
            deploy_target="vllm",
        ),
        hw=HwDecision(profile_id="generic_cpu"),
        export_format=ExportFormat.COMPRESSED_TENSORS,
    )
    stage_cfg = QuantStageConfig(name="s", backend="llm_compressor")
    art = QuantizedArtifact(
        output_dir=empty,
        manifest=ArtifactManifest(
            profile_id="generic_cpu",
            hw_decision=plan.hw,
            backend="llm_compressor",
            abstract_scheme="fp8_dynamic",
            export_format=ExportFormat.COMPRESSED_TENSORS,
        ),
    )
    ctx = PipelineContext(
        recipe={"model_id": "m"},
        recipe_path=None,
        output_dir=tmp_path,
        profile_name="generic_cpu",
        pipeline=PipelineConfig(stages=[stage_cfg]),
        stage_results=[
            StageResult(stage=stage_cfg, plan=plan, artifact=art, output_dir=empty)
        ],
    )
    with pytest.raises(QuantizedModelValidateError, match="missing config.json"):
        QuantizedModelValidateStage().run(ctx)
