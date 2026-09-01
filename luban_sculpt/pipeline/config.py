"""Pipeline configuration: multi-stage quant settings from Recipe YAML."""

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
    deploy_target: str | None = None
    ignore: list[str] | None = None
    backend_options: dict[str, Any] = Field(default_factory=dict)
    """后端私有参数；会与 recipe.quant.<backend> 合并（本字段优先）。"""

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
