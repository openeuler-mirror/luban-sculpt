"""Model architecture types and ArchQuantPolicy dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet

from pydantic import BaseModel, Field


class ModelArch(str, Enum):
    """模型架构类别：驱动 ignore / MoE / 偏好算法等（与硬件 Profile 正交）。"""

    # --- dense (non-MoE) ---
    LLAMA = "llama"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    CHATGLM = "chatglm"
    GEMMA = "gemma"
    DENSE_GENERIC = "dense_generic"

    # --- MoE ---
    QWEN_MOE = "qwen_moe"
    DEEPSEEK_MOE = "deepseek_moe"
    MIXTRAL = "mixtral"
    MOE_GENERIC = "moe_generic"

    UNKNOWN = "unknown"


MOE_ARCHES: FrozenSet[ModelArch] = frozenset(
    {
        ModelArch.QWEN_MOE,
        ModelArch.DEEPSEEK_MOE,
        ModelArch.MIXTRAL,
        ModelArch.MOE_GENERIC,
    }
)


class ModelArchSnapshot(BaseModel):
    """模型架构决议结果（resolve_model_arch 输出），供 Backend / Gate / manifest 使用。"""

    arch: ModelArch = ModelArch.UNKNOWN
    is_moe: bool = False
    is_multimodal: bool = False
    hf_model_type: str | None = None
    architectures: list[str] = Field(default_factory=list)
    source: str = "unknown"  # recipe | hf_config | heuristic | unknown
    quant_hints: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ArchQuantPolicy:
    """某 ModelArch 的静态量化默认（ignore + hints）；MoE/VL 标记在 snapshot / MOE_ARCHES。"""

    default_ignore: tuple[str, ...] = ("lm_head",)
    quant_hints: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
