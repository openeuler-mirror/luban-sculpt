"""Pipeline stages: HAE probe, per-stage compile/gate/backend (extensible)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from luban_sculpt.backends.base import BackendRouter
from luban_sculpt.compiler.recipe_compiler import compile_recipe
from luban_sculpt.hae.engine import HardwareAwareEngine
from luban_sculpt.log import get_logger
from luban_sculpt.pipeline.recipe import build_stage_recipe
from luban_sculpt.pipeline.context import PipelineContext, StageResult
from luban_sculpt.pipeline.config import QuantStageConfig
from luban_sculpt.validate.quant_capability import (
    QuantCapabilityError,
    validate_quant_capability,
)
from luban_sculpt.validate.quantized_model import (
    QuantizedModelValidateError,
    validate_quantized_model,
    write_quantized_model_report,
)
from luban_sculpt.validate.runtime import RuntimeMode

logger = get_logger(__name__)


class PipelineStage(ABC):
    """可扩展流水线阶段协议。"""

    name: str = "stage"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError


class HAEStage(PipelineStage):
    """探测硬件并加载 Profile（整条 pipeline 只跑一次）。"""

    name = "hae"

    def __init__(self, hae: HardwareAwareEngine | None = None) -> None:
        self.hae = hae

    def run(self, ctx: PipelineContext) -> PipelineContext:
        hae = self.hae or HardwareAwareEngine(ctx.profile_name)
        profile_arg = None if ctx.profile_name == "auto" else ctx.profile_name
        logger.info("HAE run profile_arg=%s", profile_arg)
        try:
            hw, probe, profile = hae.run(profile_arg)
        except Exception:
            logger.error("HAE failed profile_arg=%s", profile_arg, exc_info=True)
            raise
        ctx.hw = hw
        ctx.probe = probe
        ctx.profile = profile
        logger.info(
            "HAE done profile_id=%s vendor=%s probe_ok=%s missing_ops=%s schemes=%s",
            hw.profile_id,
            hw.vendor,
            probe.ok,
            probe.missing_ops,
            list(profile.get("schemes", {}).keys()),
        )
        if not probe.ok:
            logger.error(
                "HAE probe failed missing_ops=%s messages=%s",
                probe.missing_ops,
                probe.messages,
            )
        return ctx


class BackendComposeStage(PipelineStage):
    """按 PipelineConfig 顺序执行：compile → gate → backend.quantize。

    支持 ``input_from=previous`` 将上一阶段产物目录作为下一阶段 model_id，
    从而实现「先 llm_compressor，再 gptq / awq …」的算法×后端编配。
    """

    name = "backend_compose"

    def __init__(self, router: BackendRouter | None = None) -> None:
        self.router = router or BackendRouter()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.hw is None or ctx.probe is None:
            logger.error("BackendComposeStage requires HAEStage first")
            raise RuntimeError("HAEStage must run before BackendComposeStage")

        stages = ctx.pipeline.enabled_stages()
        if not stages:
            logger.error("pipeline has no enabled stages")
            raise ValueError("pipeline has no enabled stages")

        base_model_id = str(ctx.recipe["model_id"])
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "pipeline run %d stage(s) base_model_id=%s output_dir=%s",
            len(stages),
            base_model_id,
            ctx.output_dir,
        )

        for idx, stage in enumerate(stages):
            model_id = self._resolve_model_id(ctx, stage, base_model_id)
            stage_out = self._stage_output_dir(ctx.output_dir, stage, idx)
            stage_recipe = build_stage_recipe(ctx.recipe, stage, model_id=model_id)
            plan = compile_recipe(stage_recipe, ctx.hw, ctx.profile)
            if plan.intent.backend != stage.backend:
                plan.intent.backend = stage.backend
            if stage.algorithm:
                plan.intent.backend_options.setdefault("algo", stage.algorithm)
                plan.intent.backend_options.setdefault("algorithm", stage.algorithm)

            logger.info(
                "stage[%d] compile name=%s backend=%s scheme=%s export=%s model_id=%s",
                idx,
                stage.name,
                plan.intent.backend,
                plan.intent.abstract_scheme,
                plan.export_format.value,
                model_id,
            )
            try:
                validate_quant_capability(plan.intent, ctx.hw, ctx.profile, ctx.probe)
            except (QuantCapabilityError, ValueError) as exc:
                logger.error("stage[%d] gate failed name=%s: %s", idx, stage.name, exc)
                raise
            logger.info("stage[%d] gate passed name=%s", idx, stage.name)
            logger.info("stage[%d] quantize → %s", idx, stage_out)
            try:
                artifact = self.router.run(plan, str(stage_out))
            except Exception:
                logger.error(
                    "stage[%d] backend failed name=%s backend=%s out=%s",
                    idx,
                    stage.name,
                    plan.intent.backend,
                    stage_out,
                    exc_info=True,
                )
                raise
            result = StageResult(
                stage=stage, plan=plan, artifact=artifact, output_dir=stage_out
            )
            ctx.stage_results.append(result)
            self._write_stage_sidecar(ctx, result, idx)
            logger.info(
                "stage[%d] done name=%s manifest=%s",
                idx,
                stage.name,
                stage_out / "manifest.json",
            )

        self._write_pipeline_manifest(ctx)
        logger.info(
            "pipeline_manifest written %s",
            ctx.output_dir / "pipeline_manifest.json",
        )
        return ctx

    def _resolve_model_id(
        self, ctx: PipelineContext, stage: QuantStageConfig, base_model_id: str
    ) -> str:
        if stage.input_from == "recipe" or not ctx.stage_results:
            return base_model_id
        prev = ctx.stage_results[-1].output_dir
        return str(prev)

    def _stage_output_dir(
        self, root: Path, stage: QuantStageConfig, idx: int
    ) -> Path:
        sub = stage.output_subdir or f"stage_{idx}_{stage.backend}"
        return root / sub

    def _write_stage_sidecar(
        self, ctx: PipelineContext, result: StageResult, idx: int
    ) -> None:
        meta = {
            "index": idx,
            "name": result.stage.name,
            "backend": result.stage.backend,
            "algorithm": result.stage.algorithm,
            "abstract_scheme": result.plan.intent.abstract_scheme,
            "export_format": result.plan.export_format.value,
            "input_from": result.stage.input_from,
            "model_id": result.plan.intent.model_id,
            "output_dir": str(result.output_dir),
        }
        path = result.output_dir / "pipeline_stage.json"
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_pipeline_manifest(self, ctx: PipelineContext) -> None:
        summary: dict[str, Any] = {
            "profile_id": ctx.hw.profile_id if ctx.hw else None,
            "stages": [
                {
                    "name": r.stage.name,
                    "backend": r.stage.backend,
                    "algorithm": r.stage.algorithm,
                    "output_dir": str(r.output_dir),
                    "export_format": r.plan.export_format.value,
                    "abstract_scheme": r.plan.intent.abstract_scheme,
                }
                for r in ctx.stage_results
            ],
            "final_output": str(ctx.stage_results[-1].output_dir)
            if ctx.stage_results
            else None,
        }
        (ctx.output_dir / "pipeline_manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class QuantizedModelValidateStage(PipelineStage):
    """可选末段：对量化模型目录做 **HF config** + 可选 **vLLM runtime** 校验。

    不替代 BackendCompose 内的 ``validate_quant_capability``（compress 前检查）。
    """

    name = "quantized_model_validate"

    def __init__(
        self,
        *,
        check_hf_config: bool = True,
        check_runtime: bool = False,
        runtime_mode: RuntimeMode = "import",
    ) -> None:
        self.check_hf_config = check_hf_config
        self.check_runtime = check_runtime
        self.runtime_mode = runtime_mode

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.stage_results:
            logger.error("QuantizedModelValidateStage requires a prior compress stage")
            raise RuntimeError(
                "QuantizedModelValidateStage requires BackendComposeStage first"
            )

        last = ctx.stage_results[-1]
        model_path = last.output_dir
        logger.info(
            "quantized_model_validate start path=%s hf_config=%s runtime=%s mode=%s",
            model_path,
            self.check_hf_config,
            self.check_runtime,
            self.runtime_mode,
        )
        report = validate_quantized_model(
            model_path,
            plan=last.plan,
            check_hf_config=self.check_hf_config,
            check_runtime=self.check_runtime,
            runtime_mode=self.runtime_mode,
        )
        report_path = write_quantized_model_report(model_path, report)
        if not report["ok"]:
            for err in report.get("errors") or []:
                logger.error("quantized_model_validate: %s", err)
            raise QuantizedModelValidateError(
                "; ".join(report.get("errors") or ["failed"])
            )
        logger.info(
            "quantized_model_validate ok path=%s report=%s", model_path, report_path
        )
        return ctx
