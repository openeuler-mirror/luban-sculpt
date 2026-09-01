"""海光 DCU / enginex-hygon-vllm：压缩规格与推理 sidecar。

压缩仍委托 llm-compressor（GPTQ/AWQ/W8A8 等最接近路径）；
推理 ``--quantization`` 与 lmslim/lightop 约定放在本包。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from luban_sculpt.backends.llm_compressor.scheme_map import CompressSpec


@dataclass(frozen=True)
class HygonCompressSpec:
    """海光推理方法 + 对应 LC 压缩规格。"""

    compress: CompressSpec
    infer_quantization: str
    note: str = ""


# abstract_scheme → 海光推理方法 + LC 压缩基线
_HYGON_SCHEMES: dict[str, HygonCompressSpec] = {
    "slimquant_w4a8": HygonCompressSpec(
        compress=CompressSpec(
            algorithm="gptq",
            scheme="W4A16",
            block_size=128,
            quantization_format="pack-quantized",
        ),
        infer_quantization="slimquant_w4a8",
        note=(
            "LC 产出 W4A16 CT；DCU 上 --quantization slimquant_w4a8 前 "
            "通常还需 lmslim W4A8 pack/repack。"
        ),
    ),
    "slimquant_w4a8_marlin": HygonCompressSpec(
        compress=CompressSpec(
            algorithm="gptq",
            scheme="W4A16",
            block_size=128,
            quantization_format="pack-quantized",
        ),
        infer_quantization="slimquant_w4a8_marlin",
        note="同上；decode 高吞吐走 lightop Marlin 布局。",
    ),
    "blockwise_int8": HygonCompressSpec(
        compress=CompressSpec(
            algorithm="quantization",
            scheme="W8A8",
            quantization_format=None,
        ),
        infer_quantization="blockwise_int8",
        note="W8A8 基线；推理加 --weight-block-size 128 128。",
    ),
    "awq_marlin": HygonCompressSpec(
        compress=CompressSpec(
            algorithm="awq",
            scheme="W4A16_ASYM",
            quantization_format="pack-quantized",
        ),
        infer_quantization="awq_marlin",
        note="AWQ 权重；DCU 上 awq_marlin kernel 由 lightop 提供。",
    ),
}


def is_hygon_scheme(abstract_scheme: str) -> bool:
    return abstract_scheme in _HYGON_SCHEMES


def list_hygon_schemes() -> list[str]:
    return sorted(_HYGON_SCHEMES)


def resolve_hygon_spec(
    abstract_scheme: str,
    *,
    override: dict[str, Any] | None = None,
) -> HygonCompressSpec:
    """解析海光 scheme；YAML ``hygon:`` / ``llm_compressor:`` 可覆盖 compress 字段。"""
    if abstract_scheme not in _HYGON_SCHEMES:
        raise KeyError(
            f"unknown hygon scheme {abstract_scheme!r}; "
            f"known={list_hygon_schemes()}"
        )
    base = _HYGON_SCHEMES[abstract_scheme]
    ov = override or {}
    # 允许 nested llm_compressor 覆盖（与旧 recipe 兼容）
    lc = ov.get("llm_compressor") if isinstance(ov.get("llm_compressor"), dict) else ov
    c = base.compress
    algo = str(lc.get("algorithm") or c.algorithm).lower()
    if algo not in ("gptq", "awq", "quantization"):
        algo = c.algorithm
    compress = CompressSpec(
        algorithm=algo,
        scheme=str(lc.get("scheme") or c.scheme),
        block_size=lc.get("block_size", c.block_size),
        quantization_format=lc.get("quantization_format", c.quantization_format),
    )
    return HygonCompressSpec(
        compress=compress,
        infer_quantization=str(
            ov.get("infer_quantization") or base.infer_quantization
        ),
        note=str(ov.get("note") or base.note),
    )


def hygon_infer_extra_args(infer_quantization: str) -> list[str]:
    if infer_quantization == "blockwise_int8":
        return ["--weight-block-size", "128", "128"]
    return []


def build_hygon_infer_payload(
    save_dir: str,
    infer_quantization: str,
    *,
    note: str = "",
) -> dict[str, Any]:
    """enginex-hygon-vllm 启动 sidecar（对齐 llama3_example）。"""
    extra = hygon_infer_extra_args(infer_quantization)
    serve = f"vllm serve {save_dir} --quantization {infer_quantization}"
    if extra:
        serve = serve + " " + " ".join(extra)
    return {
        "engine": "enginex-hygon-vllm",
        "based_on": "vllm==0.9.2",
        "quantization": infer_quantization,
        "extra_args": extra,
        "env": {
            "W8A8_SUPPORT_METHODS": "1",
            "USE_FUSED_RMS_QUANT": "1",
        },
        "pip_required": ["lmslim", "lightop"],
        "serve_example": serve,
        "note": note
        or (
            "slimquant_* 若无法直接加载本目录 CT/AWQ 权重，"
            "需在 DTK 容器用 lmslim/lightop 做 W4A8 repack 后再 serve。"
        ),
    }


def lc_options_from_hygon(
    hygon_spec: HygonCompressSpec,
    hygon_opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把海光 CompressSpec 转成 llm_compressor runner 可用的 options。"""
    opts = dict(hygon_opts or {})
    c = hygon_spec.compress
    lc = {
        "algorithm": c.algorithm,
        "scheme": c.scheme,
        "trust_remote_code": opts.get("trust_remote_code", True),
    }
    if c.block_size is not None:
        lc["block_size"] = c.block_size
    if c.quantization_format is not None:
        lc["quantization_format"] = c.quantization_format
    # 透传 oneshot / modifiers 等
    for key in ("oneshot", "modifiers", "targets", "ignore"):
        if key in opts:
            lc[key] = opts[key]
    return lc
