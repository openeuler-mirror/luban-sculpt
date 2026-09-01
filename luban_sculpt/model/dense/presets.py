"""Dense (non-MoE) arch quant policies."""

from __future__ import annotations

from luban_sculpt.model.types import ArchQuantPolicy, ModelArch

DENSE_PRESETS: dict[ModelArch, ArchQuantPolicy] = {
    ModelArch.LLAMA: ArchQuantPolicy(
        default_ignore=("lm_head",),
        notes="Dense Llama; standard lm_head skip",
    ),
    ModelArch.QWEN: ArchQuantPolicy(
        default_ignore=("lm_head", "re:.*mlp.gate$"),
        notes="Qwen dense: keep mlp.gate in higher precision when possible",
    ),
    ModelArch.DEEPSEEK: ArchQuantPolicy(
        default_ignore=("lm_head",),
        notes="DeepSeek dense",
    ),
    ModelArch.MISTRAL: ArchQuantPolicy(
        default_ignore=("lm_head",),
    ),
    ModelArch.CHATGLM: ArchQuantPolicy(
        default_ignore=("lm_head", "transformer.output_layer"),
        notes="ChatGLM/GLM output_layer often skipped",
    ),
    ModelArch.GEMMA: ArchQuantPolicy(
        default_ignore=("lm_head",),
    ),
    ModelArch.DENSE_GENERIC: ArchQuantPolicy(
        default_ignore=("lm_head",),
        notes="Fallback for baichuan / yi / phi / internlm / llava etc.",
    ),
    ModelArch.UNKNOWN: ArchQuantPolicy(
        default_ignore=("lm_head",),
        notes="Unresolved arch; only safe lm_head skip",
    ),
}
