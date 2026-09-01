"""Composable quantization pipeline (HAE → multi backend/algorithm stages).

Prefer::

    from luban_sculpt.pipeline import QuantPipeline

    pipe = QuantPipeline(profile_name="auto")
    artifact = pipe.run(recipe_path, output_dir)

Multi-stage recipe example::

    pipeline:
      stages:
        - name: fp8
          backend: llm_compressor
          abstract_scheme: fp8_dynamic
        - name: awq
          backend: awq
          abstract_scheme: hygon_w4a16_awq
          input_from: previous
"""

from luban_sculpt.pipeline.recipe import build_stage_recipe, parse_pipeline_config
from luban_sculpt.pipeline.context import PipelineContext, StageResult
from luban_sculpt.pipeline.runner import QuantPipeline
from luban_sculpt.pipeline.config import PipelineConfig, QuantStageConfig
from luban_sculpt.pipeline.stages import (
    BackendComposeStage,
    HAEStage,
    PipelineStage,
    QuantizedModelValidateStage,
)

__all__ = [
    "BackendComposeStage",
    "HAEStage",
    "PipelineConfig",
    "PipelineContext",
    "PipelineStage",
    "QuantPipeline",
    "QuantStageConfig",
    "QuantizedModelValidateStage",
    "StageResult",
    "build_stage_recipe",
    "parse_pipeline_config",
]
