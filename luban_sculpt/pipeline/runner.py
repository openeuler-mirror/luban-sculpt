"""QuantPipeline runner: composable HAE → multi backend/algorithm stages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from luban_sculpt.backends.base import BackendRouter
from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml
from luban_sculpt.contracts import QuantizedArtifact
from luban_sculpt.hae.engine import HardwareAwareEngine
from luban_sculpt.log import configure_logging, get_logger
from luban_sculpt.pipeline.recipe import parse_pipeline_config
from luban_sculpt.pipeline.context import PipelineContext
from luban_sculpt.pipeline.stages import (
    QuantizedModelValidateStage,
    BackendComposeStage,
    HAEStage,
    PipelineStage,
)
from luban_sculpt.validate.runtime import RuntimeMode

logger = get_logger(__name__)


class QuantPipeline:
    """可扩展量化流水线。

    默认阶段：``HAEStage`` → ``BackendComposeStage``。
    设置 ``validate_quantized_model=True`` 时追加 ``QuantizedModelValidateStage``
    （HF 配置校验；``validate_runtime=True`` 时再做运行时校验）。
    也可通过 ``stages=`` 完全替换阶段链。
    """

    def __init__(
        self,
        profile_name: str = "auto",
        *,
        router: BackendRouter | None = None,
        hae: HardwareAwareEngine | None = None,
        stages: Sequence[PipelineStage] | None = None,
        validate_quantized_model: bool = False,
        validate_runtime: bool = False,
        runtime_mode: RuntimeMode = "import",
    ) -> None:
        self.profile_name = profile_name
        self.router = router or BackendRouter()
        self.hae = hae or HardwareAwareEngine(profile_name)
        configure_logging(level=os.environ.get("LUBAN_LOG_LEVEL", "info"))

        if stages is not None:
            self.stages: list[PipelineStage] = list(stages)
        else:
            self.stages = [
                HAEStage(self.hae),
                BackendComposeStage(self.router),
            ]
            if validate_quantized_model:
                self.stages.append(
                    QuantizedModelValidateStage(
                        check_hf_config=True,
                        check_runtime=validate_runtime,
                        runtime_mode=runtime_mode,
                    )
                )

    def run(self, recipe_path: Path, output_dir: Path) -> QuantizedArtifact:
        """执行流水线，返回**最后一阶段**产物。"""
        recipe = load_recipe_yaml(recipe_path)
        return self.run_recipe(recipe, output_dir, recipe_path=recipe_path)

    def run_recipe(
        self,
        recipe: dict,
        output_dir: Path,
        *,
        recipe_path: Path | None = None,
    ) -> QuantizedArtifact:
        pipeline_config = parse_pipeline_config(recipe)
        logger.info(
            "pipeline start profile=%s recipe=%s model_id=%s stages=%s output=%s",
            self.profile_name,
            recipe_path,
            recipe.get("model_id"),
            [s.name for s in self.stages],
            output_dir,
        )
        ctx = PipelineContext(
            recipe=recipe,
            recipe_path=recipe_path,
            output_dir=output_dir,
            profile_name=self.profile_name,
            pipeline=pipeline_config,
        )
        try:
            for stage in self.stages:
                logger.info("pipeline enter stage=%s", stage.name)
                ctx = stage.run(ctx)
                logger.info("pipeline leave stage=%s", stage.name)
            if not ctx.last_artifact:
                raise RuntimeError("pipeline produced no artifact")
        except Exception:
            logger.error(
                "pipeline failed profile=%s recipe=%s output=%s",
                self.profile_name,
                recipe_path,
                output_dir,
                exc_info=True,
            )
            raise
        logger.info(
            "pipeline done final_output=%s backend=%s scheme=%s export=%s",
            ctx.last_artifact.output_dir,
            ctx.last_artifact.manifest.backend,
            ctx.last_artifact.manifest.abstract_scheme,
            ctx.last_artifact.manifest.export_format.value,
        )
        return ctx.last_artifact
