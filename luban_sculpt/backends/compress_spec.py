"""Recipe ``abstract_scheme`` → llm-compressor ``CompressSpec``（backends 层公共模块）。

与 upstream ``examples/quantization_*`` 对齐；海光等 vendor 扩展见
``luban_sculpt.backends.hygon.check``。

``fp8_dynamic`` / ``nvfp4`` 与 ``w8a8_fp8`` / ``w4a4_fp4`` 并列入库（同 LC scheme），
不再单独维护别名表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# abstract_scheme → llm-compressor ``scheme=`` 字符串
_ABSTRACT_TO_SCHEME: dict[str, str] = {
    "w4a16": "W4A16",
    "w4a16_fp4": "NVFP4A16",
    "w4a4_fp4": "NVFP4",
    "nvfp4": "NVFP4",
    "w4a4_mxfp4": "MXFP4",
    "w4a8_fp8": "W4AFP8",
    "w8a8_fp8": "FP8_DYNAMIC",
    "fp8_dynamic": "FP8_DYNAMIC",
    "w8a8_int8": "W8A8",
    "w8a8_mxfp8": "MXFP8",
    "fp8_block": "FP8_BLOCK",
}

# 未写 llm_compressor.algorithm 时默认走 GPTQ 的 abstract_scheme
_GPTQ_DEFAULT_ABSTRACT = frozenset({"w4a16", "w8a8_int8", "w4a8_fp8"})

# GPTQ 且未写 block_size 时默认 128
_W4A16_BLOCK128 = frozenset({"w4a16"})


@dataclass(frozen=True)
class CompressSpec:
    """压缩侧规格（llm_compressor / hygon / modifier 链共用）。"""

    algorithm: str
    scheme: str
    block_size: int | None = None
    quantization_format: str | None = None


def scheme_for_abstract(abstract_scheme: str) -> str | None:
    """abstract_scheme → llm-compressor ``scheme`` 字符串；未知则 None。"""
    return _ABSTRACT_TO_SCHEME.get(abstract_scheme)


def resolve_compress_spec(
    abstract_scheme: str,
    *,
    compressor_options: dict[str, Any] | None = None,
) -> CompressSpec:
    """由 abstract_scheme（及可选 llm_compressor YAML 覆盖）得到 CompressSpec。"""
    options = compressor_options or {}
    if options.get("scheme") and options.get("algorithm"):
        algo = str(options["algorithm"]).lower()
        if algo not in ("gptq", "awq", "quantization"):
            algo = "quantization"
        return CompressSpec(
            algorithm=algo,
            scheme=str(options["scheme"]),
            block_size=options.get("block_size"),
            quantization_format=options.get("quantization_format"),
        )

    scheme = (
        options.get("scheme")
        or scheme_for_abstract(abstract_scheme)
        or "FP8_DYNAMIC"
    )
    algo = str(options.get("algorithm") or "quantization").lower()
    if algo not in ("gptq", "awq", "quantization"):
        algo = "quantization"
    if abstract_scheme in _GPTQ_DEFAULT_ABSTRACT and "algorithm" not in options:
        algo = "gptq"

    block_size = options.get("block_size")
    if block_size is None and algo == "gptq" and abstract_scheme in _W4A16_BLOCK128:
        block_size = 128

    quant_fmt = options.get("quantization_format")
    if quant_fmt is None and algo in ("gptq", "awq"):
        quant_fmt = "pack-quantized"

    return CompressSpec(
        algorithm=algo,
        scheme=str(scheme),
        block_size=block_size,
        quantization_format=quant_fmt,
    )
