"""Parse Recipe YAML into PipelineConfig and build per-stage recipe views."""

from __future__ import annotations

from typing import Any

from luban_sculpt.pipeline.config import PipelineConfig, QuantStageConfig


def parse_pipeline_config(recipe: dict[str, Any]) -> PipelineConfig:
    """从 recipe 解析流水线设置。

    支持三种写法（优先级从高到低）：

    1. ``pipeline.stages: [...]`` 显式多阶段
    2. ``pipeline: [ {...}, ... ]`` 列表简写
    3. 仅 ``quant:`` → 单阶段（兼容现有 Recipe）
    """
    raw = recipe.get("pipeline")
    if isinstance(raw, dict) and raw.get("stages") is not None:
        stages = [_parse_stage(item, idx) for idx, item in enumerate(raw["stages"])]
        return PipelineConfig(stages=stages)
    if isinstance(raw, list):
        stages = [_parse_stage(item, idx) for idx, item in enumerate(raw)]
        return PipelineConfig(stages=stages)

    q = recipe.get("quant") or {}
    backend = str(q.get("backend", "llm_compressor"))
    # quant.<backend> 专属配置块，如 quant.gptq: { bits: 4 }
    backend_cfg = q.get(backend) if isinstance(q.get(backend), dict) else {}
    opts = dict(q.get("backend_options") or {})
    if backend_cfg:
        opts = {**opts, **backend_cfg}
    algo = opts.get("algo") or opts.get("algorithm") or q.get("algorithm") or q.get("algo")
    return PipelineConfig(
        stages=[
            QuantStageConfig(
                name="default",
                backend=backend,
                algorithm=str(algo) if algo else None,
                abstract_scheme=q.get("abstract_scheme") or q.get("scheme"),
                deploy_target=q.get("deploy_target"),
                ignore=list(q.get("ignore") or []) or None,
                backend_options=opts,
                input_from="recipe",
                output_subdir=None,
            )
        ]
    )


def _parse_stage(item: Any, idx: int) -> QuantStageConfig:
    if not isinstance(item, dict):
        raise TypeError(f"pipeline stage[{idx}] must be a mapping, got {type(item)}")
    data = dict(item)
    data.setdefault("name", f"stage_{idx}")
    if "backend" not in data:
        raise ValueError(f"pipeline stage[{idx}] missing required field 'backend'")
    if data.get("algorithm") is None and data.get("algo") is not None:
        data["algorithm"] = data.pop("algo")
    else:
        data.pop("algo", None)
    return QuantStageConfig.model_validate(data)


def build_stage_recipe(
    base_recipe: dict[str, Any],
    stage: QuantStageConfig,
    *,
    model_id: str,
) -> dict[str, Any]:
    """将阶段设置合并进 recipe 视图，供 ``compile_recipe`` 使用。"""
    base_q = dict(base_recipe.get("quant") or {})
    opts = dict(stage.backend_options or {})
    # inherit quant.<backend> from base recipe when stage did not override
    base_backend_cfg = base_q.get(stage.backend)
    if isinstance(base_backend_cfg, dict):
        opts = {**base_backend_cfg, **opts}
    if stage.algorithm:
        opts.setdefault("algo", stage.algorithm)
        opts.setdefault("algorithm", stage.algorithm)

    quant: dict[str, Any] = {
        "backend": stage.backend,
        "abstract_scheme": stage.abstract_scheme
        or base_q.get("abstract_scheme")
        or base_q.get("scheme")
        or "fp8_dynamic",
        "deploy_target": stage.deploy_target
        or base_q.get("deploy_target")
        or "vllm_cuda",
        "ignore": (
            list(stage.ignore)
            if stage.ignore is not None
            else list(base_q.get("ignore") or [])
        ),
        stage.backend: opts,
    }

    out = {
        k: v
        for k, v in base_recipe.items()
        if k not in {"quant", "pipeline"}
    }
    out["model_id"] = model_id
    out["quant"] = quant
    return out
