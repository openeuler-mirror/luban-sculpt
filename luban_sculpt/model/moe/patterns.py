"""MoE HF / architecture / model_id match patterns (checked before dense)."""

from __future__ import annotations

import re

from luban_sculpt.model.types import ModelArch

HF_MODEL_TYPE_MAP: dict[str, ModelArch] = {
    "mixtral": ModelArch.MIXTRAL,
    "qwen2_moe": ModelArch.QWEN_MOE,
    "qwen3_moe": ModelArch.QWEN_MOE,
    "deepseek_v2": ModelArch.DEEPSEEK_MOE,
    "deepseek_v3": ModelArch.DEEPSEEK_MOE,
}

ARCH_PATTERNS: list[tuple[re.Pattern[str], ModelArch]] = [
    (re.compile(r"Qwen2Moe|Qwen3Moe", re.I), ModelArch.QWEN_MOE),
    (re.compile(r"DeepseekV[23]|DeepSeekMoe", re.I), ModelArch.DEEPSEEK_MOE),
    (re.compile(r"Mixtral", re.I), ModelArch.MIXTRAL),
]

ID_PATTERNS: list[tuple[re.Pattern[str], ModelArch]] = [
    (re.compile(r"qwen3?[\W_]*moe|moe[\W_]*qwen", re.I), ModelArch.QWEN_MOE),
    (re.compile(r"deepseek[\W_]*v[23]|deepseek[\W_]*moe", re.I), ModelArch.DEEPSEEK_MOE),
    (re.compile(r"mixtral", re.I), ModelArch.MIXTRAL),
    (re.compile(r"[\W_]moe[\W_]|moe-", re.I), ModelArch.MOE_GENERIC),
]
