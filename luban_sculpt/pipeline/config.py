"""Pipeline configuration and recipe parsing for multi-stage quant."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


InputFrom = Literal["recipe", "previous"]


class QuantStageConfig(BaseModel):
    """单段量化设置：backend、算法与阶段间串接。"""

    name: str = "stage"
    backend: str
    """量化后端 entry_point 名，如 llm_compressor / gptq / awq。"""

    algorithm: str | None = None
    """算法提示（写入 backend_options.algo），如 awq / gptq / smoothquant。"""

    abstract_scheme: str | None = None
    ignore: list[str] | None = None
    backend_options: dict[str, Any] = Field(default_factory=dict)
    """后端私有参数；可与 stage 上的 ``<backend>: { ... }`` 块合并（本字段优先）。"""

    input_from: InputFrom = "recipe"
    """recipe=原始 model_id；previous=上一阶段 output_dir。"""

    output_subdir: str | None = None
    """相对总 output 的子目录；默认 stage_{i}_{backend}。"""

    enabled: bool = True


class PipelineConfig(BaseModel):
    """多阶段流水线设置（可扩展）。"""

    stages: list[QuantStageConfig] = Field(default_factory=list)

    def enabled_stages(self) -> list[QuantStageConfig]:
        return [s for s in self.stages if s.enabled]


def parse_pipeline_config(recipe: dict[str, Any]) -> PipelineConfig:
    """从 recipe 解析 ``PipelineConfig``（``pipeline.stages``，单阶段也写 1 个元素）。"""
    if recipe.get("quant") is not None:
        raise ValueError(
            "top-level 'quant' is removed; use pipeline.stages (see recipes/*.yaml)"
        )
    stage_items = _collect_stage_mappings(recipe)
    stages = [_parse_stage(item, idx) for idx, item in enumerate(stage_items)]
    return PipelineConfig(stages=stages)


def _collect_stage_mappings(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    raw = recipe.get("pipeline")
    if raw is None:
        raise ValueError("recipe requires pipeline.stages")

    if isinstance(raw, dict):
        if raw.get("stages") is not None:
            return list(raw["stages"])
        raise ValueError(
            "recipe.pipeline must be a mapping with 'stages: [...]' "
            f"(got keys {sorted(raw.keys())!r})"
        )

    if isinstance(raw, list):
        raise ValueError(
            "pipeline as a bare list is not supported; use pipeline.stages: [...]"
        )

    raise ValueError(
        f"recipe.pipeline must be a mapping with 'stages', got {type(raw).__name__!r}"
    )


def _merge_backend_block(
    backend_options: dict[str, Any] | None,
    recipe_block: Any,
) -> dict[str, Any]:
    """``backend_options`` 与 ``<backend>: { ... }`` 合并，专属块覆盖同名键。"""
    opts = dict(backend_options or {})
    if isinstance(recipe_block, dict):
        opts = {**opts, **recipe_block}
    return opts


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

    backend = str(data["backend"])
    backend_cfg = data.pop(backend, None)
    data["backend_options"] = _merge_backend_block(
        data.get("backend_options"), backend_cfg
    )
    if data.get("algorithm") is None:
        opts = data["backend_options"]
        algo = opts.get("algo") or opts.get("algorithm")
        if algo:
            data["algorithm"] = str(algo)
    if data.get("ignore") == []:
        data["ignore"] = None
    return QuantStageConfig.model_validate(data)


def build_stage_recipe(
    base_recipe: dict[str, Any],
    stage: QuantStageConfig,
    *,
    model_id: str,
) -> dict[str, Any]:
    """将单个 stage 合成带 ``quant`` 的 recipe 视图，供 ``compile_recipe`` 使用。"""
    opts = dict(stage.backend_options or {})
    if stage.algorithm:
        opts.setdefault("algo", stage.algorithm)
        opts.setdefault("algorithm", stage.algorithm)

    quant: dict[str, Any] = {
        "backend": stage.backend,
        "abstract_scheme": stage.abstract_scheme or "fp8_dynamic",
        "ignore": list(stage.ignore) if stage.ignore is not None else [],
        stage.backend: opts,
    }

    out = {
        k: v
        for k, v in base_recipe.items()
        if k not in {"pipeline"}
    }
    out["model_id"] = model_id
    out["quant"] = quant
    return out
