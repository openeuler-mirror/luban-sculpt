"""Dense (non-MoE) HF / architecture / model_id match patterns."""

from __future__ import annotations

import re

from luban_sculpt.model.types import ModelArch

_GENERIC = ModelArch.DENSE_GENERIC

# HF config.model_type → arch (unlisted families → dense_generic)
HF_MODEL_TYPE_MAP: dict[str, ModelArch] = {
    "llama": ModelArch.LLAMA,
    "mistral": ModelArch.MISTRAL,
    "qwen2": ModelArch.QWEN,
    "qwen3": ModelArch.QWEN,
    "qwen2_vl": ModelArch.QWEN,
    "qwen2_5_vl": ModelArch.QWEN,
    "deepseek": ModelArch.DEEPSEEK,
    "chatglm": ModelArch.CHATGLM,
    "chatglm2": ModelArch.CHATGLM,
    "glm": ModelArch.CHATGLM,
    "glm4": ModelArch.CHATGLM,
    "gemma": ModelArch.GEMMA,
    "gemma2": ModelArch.GEMMA,
    "baichuan": _GENERIC,
    "yi": _GENERIC,
    "phi": _GENERIC,
    "phi3": _GENERIC,
    "internlm": _GENERIC,
    "internlm2": _GENERIC,
}

# HuggingFace config.json → architectures[] 里的 *ForCausalLM 类名（按顺序，先命中先用）。
# 例：["Qwen2ForCausalLM"] → qwen；["LlavaForConditionalGeneration"] → dense_generic。
# resolve 里在 model_type 未命中 HF_MODEL_TYPE_MAP 时才扫这条链（MoE 规则在 moe/patterns 里更靠前）。
ARCH_PATTERNS: list[tuple[re.Pattern[str], ModelArch]] = [
    (re.compile(r"Qwen2VL|Qwen2_5_VL|Qwen2|Qwen3|QWen", re.I), ModelArch.QWEN),
    (re.compile(r"Deepseek|DeepSeek", re.I), ModelArch.DEEPSEEK),
    (re.compile(r"Llama|LlamaForCausalLM", re.I), ModelArch.LLAMA),
    (re.compile(r"Mistral", re.I), ModelArch.MISTRAL),
    (re.compile(r"ChatGLM|GlmFor", re.I), ModelArch.CHATGLM),
    (re.compile(r"Gemma", re.I), ModelArch.GEMMA),
    (re.compile(r"Llava|InternVL", re.I), _GENERIC),
    (re.compile(r"Baichuan|InternLM|Phi", re.I), _GENERIC),
]

# Hub slug / 本地目录路径字符串（顺序敏感：qwen-vl 要在 qwen 前，否则会误判成纯 text qwen）。
# resolve 在缺 config.json 或 arch 仍未定时的最后一档启发式。
ID_PATTERNS: list[tuple[re.Pattern[str], ModelArch]] = [
    (re.compile(r"qwen.*vl|vl.*qwen", re.I), ModelArch.QWEN),
    (re.compile(r"llava|internvl", re.I), _GENERIC),
    (re.compile(r"qwen", re.I), ModelArch.QWEN),
    (re.compile(r"deepseek", re.I), ModelArch.DEEPSEEK),
    (re.compile(r"llama|meta-llama", re.I), ModelArch.LLAMA),
    (re.compile(r"mistral", re.I), ModelArch.MISTRAL),
    (re.compile(r"chatglm|glm-?\d", re.I), ModelArch.CHATGLM),
    (re.compile(r"gemma", re.I), ModelArch.GEMMA),
    (re.compile(r"baichuan|internlm|phi-?\d|\byi-", re.I), _GENERIC),
]


def match_arch_from_hf_architecture(class_name: str) -> ModelArch | None:
    """对 architectures[] 中单条类名做 ARCH_PATTERNS 匹配（与 resolve 一致）。"""
    for pat, arch in ARCH_PATTERNS:
        if pat.search(class_name):
            return arch
    return None


def match_arch_from_model_id(model_id: str) -> ModelArch | None:
    """对 model_id 全路径 + 最后一段做 ID_PATTERNS 匹配（与 resolve 一致）。"""
    name = model_id.replace("\\", "/")
    base = name.rsplit("/", 1)[-1]
    blob = f"{name} {base}"
    for pat, arch in ID_PATTERNS:
        if pat.search(blob):
            return arch
    return None


# fmt: off
# (kind, input, expected) — kind: "arch" | "id"
_PATTERN_EXAMPLES: list[tuple[str, str, ModelArch]] = [
    # ARCH_PATTERNS：读本地 weights 时 config.json 里 architectures 的典型类名
    ("arch", "Qwen2ForCausalLM", ModelArch.QWEN),
    ("arch", "Meta-Llama-3-8B — class LlamaForCausalLM", ModelArch.LLAMA),
    ("arch", "LlavaForConditionalGeneration", ModelArch.DENSE_GENERIC),
    ("arch", "BaichuanForCausalLM", ModelArch.DENSE_GENERIC),
    # ID_PATTERNS：只有 Hub 名 / 路径、还没有 config 时
    ("id", "Qwen/Qwen2.5-7B-Instruct", ModelArch.QWEN),
    ("id", "Qwen/Qwen2-VL-7B-Instruct", ModelArch.QWEN),
    ("id", "llava-hf/llava-1.5-7b-hf", ModelArch.DENSE_GENERIC),
    ("id", "meta-llama/Llama-3.1-8B-Instruct", ModelArch.LLAMA),
    ("id", "baichuan-inc/Baichuan2-7B-Chat", ModelArch.DENSE_GENERIC),
]
# fmt: on


def _run_pattern_examples() -> None:
    for kind, text, expected in _PATTERN_EXAMPLES:
        if kind == "arch":
            got = match_arch_from_hf_architecture(text)
        else:
            got = match_arch_from_model_id(text)
        assert got == expected, f"{kind} {text!r}: got {got}, want {expected}"


if __name__ == "__main__":
    _run_pattern_examples()
    print(f"OK: {len(_PATTERN_EXAMPLES)} dense pattern examples")
