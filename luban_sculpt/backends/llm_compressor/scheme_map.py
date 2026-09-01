"""abstract_scheme → llm-compressor compress recipe（通用，不含海光专属）。

海光 slimquant_* / blockwise_int8 / awq_marlin 见 ``luban_sculpt.backends.hygon``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompressSpec:
    """压缩侧规格（recipe_builder / runner 共用）。"""

    # quantization | gptq | awq
    algorithm: str
    scheme: str
    block_size: int | None = None
    # save_pretrained(quantization_format=...)；None 表示不传
    quantization_format: str | None = None


# 通用 LC scheme 名映射（默认走 QuantizationModifier）
_SCHEME_LC: dict[str, str] = {
    "fp8_dynamic": "FP8_DYNAMIC",
    "fp8_block": "FP8_BLOCK",
    "w8a8_int8": "W8A8",
    "w4a16_g32": "W4A16",
    "w4a16": "W4A16",
    "nvfp4": "NVFP4",
    "compressed-tensors": "W4A16",
}


def resolve_compress_spec(
    abstract_scheme: str,
    *,
    lc_override: dict[str, Any] | None = None,
) -> CompressSpec:
    """由 abstract_scheme（及可选 llm_compressor YAML 覆盖）得到 CompressSpec。"""
    lc = lc_override or {}
    # 若 YAML/上游已注入 algorithm+scheme（如 hygon backend），优先用之
    if lc.get("scheme") and lc.get("algorithm"):
        algo = str(lc["algorithm"]).lower()
        if algo not in ("gptq", "awq", "quantization"):
            algo = "quantization"
        return CompressSpec(
            algorithm=algo,
            scheme=str(lc["scheme"]),
            block_size=lc.get("block_size"),
            quantization_format=lc.get("quantization_format"),
        )

    scheme = lc.get("scheme") or _SCHEME_LC.get(abstract_scheme, "FP8_DYNAMIC")
    algo = str(lc.get("algorithm") or "quantization").lower()
    if algo not in ("gptq", "awq", "quantization"):
        algo = "quantization"
    # compressed-tensors / w4a16 默认 GPTQ W4A16 + pack-quantized
    if abstract_scheme in ("compressed-tensors", "w4a16", "w4a16_g32") and "algorithm" not in lc:
        algo = "gptq"
    return CompressSpec(
        algorithm=algo,
        scheme=str(scheme),
        block_size=lc.get("block_size")
        or (128 if algo == "gptq" and abstract_scheme in ("w4a16", "w4a16_g32", "compressed-tensors") else None),
        quantization_format=lc.get("quantization_format")
        or ("pack-quantized" if algo in ("gptq", "awq") else None),
    )
