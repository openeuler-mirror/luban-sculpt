"""YAML Recipe → QuantIntent + BackendPlan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent
from luban_sculpt.hae.profile_fields import expected_infer_runtime
from luban_sculpt.log import get_logger
from luban_sculpt.hae.resolve_quant import resolve_quant_block
from luban_sculpt.model import (
    apply_arch_to_backend_options,
    merge_ignore_list,
    resolve_model_arch,
)
from luban_sculpt.compiler.recipe_template import resolve_recipe_extends
from luban_sculpt.model.recipe_model import normalize_recipe_model

logger = get_logger(__name__)

_BACKEND_EXPORT: dict[str, ExportFormat] = {
    "llm_compressor": ExportFormat.COMPRESSED_TENSORS,
    "hygon": ExportFormat.COMPRESSED_TENSORS,
    "infera": ExportFormat.COMPRESSED_TENSORS,
    "custom": ExportFormat.COMPRESSED_TENSORS,
    "msmodelslim": ExportFormat.VLLM_ASCEND,
    "awq": ExportFormat.AWQ_HF,
    "gptq": ExportFormat.GPTQ_HF,
}

_SCHEME_EXPORT_OVERRIDE: dict[str, ExportFormat] = {
    # llm-compressor examples/quantization_*
    "w4a16": ExportFormat.COMPRESSED_TENSORS,
    "w4a16_fp4": ExportFormat.COMPRESSED_TENSORS,
    "w4a4_fp4": ExportFormat.COMPRESSED_TENSORS,
    "w4a4_mxfp4": ExportFormat.COMPRESSED_TENSORS,
    "w4a8_fp8": ExportFormat.COMPRESSED_TENSORS,
    "w8a8_fp8": ExportFormat.COMPRESSED_TENSORS,
    "w8a8_int8": ExportFormat.COMPRESSED_TENSORS,
    "w8a8_mxfp8": ExportFormat.COMPRESSED_TENSORS,
    "fp8_dynamic": ExportFormat.COMPRESSED_TENSORS,
    "fp8_block": ExportFormat.COMPRESSED_TENSORS,
    "nvfp4": ExportFormat.COMPRESSED_TENSORS,
    "fp8_hf": ExportFormat.FP8_HF,
    "ascend_w8a8": ExportFormat.VLLM_ASCEND,
    "ascend_w4a8": ExportFormat.VLLM_ASCEND,
    "ascend_fp8": ExportFormat.VLLM_ASCEND,
    "ascend_w8a16": ExportFormat.VLLM_ASCEND,
    "w4_gptq": ExportFormat.GPTQ_HF,
    "w4a16_awq": ExportFormat.AWQ_HF,
    "hygon_w4a16_awq": ExportFormat.AWQ_HF,
    "hygon_w8a8_gptq": ExportFormat.GPTQ_HF,
    "slimquant_w4a8": ExportFormat.COMPRESSED_TENSORS,
    "slimquant_w4a8_marlin": ExportFormat.COMPRESSED_TENSORS,
    "blockwise_int8": ExportFormat.COMPRESSED_TENSORS,
    "awq_marlin": ExportFormat.COMPRESSED_TENSORS,
}


def load_recipe_yaml(path: Path, *, validate_model_layout: bool = True) -> dict[str, Any]:
    """读取 recipe YAML 并规范化 ``model`` 块 → ``model_id``。"""
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"recipe must be a mapping: {path}")
    doc = resolve_recipe_extends(doc, recipe_path=path)
    return normalize_recipe_model(doc, validate_layout=validate_model_layout)


def _scale_calib_for_arch(calib: dict[str, Any], arch_snapshot) -> dict[str, Any]:
    """MoE 等架构可按 quant_hints.calib_samples_factor 放大校准样本。"""
    out = dict(calib or {})
    factor = arch_snapshot.quant_hints.get("calib_samples_factor")
    if factor and "max_samples" in out:
        try:
            out["max_samples"] = max(1, int(int(out["max_samples"]) * float(factor)))
        except (TypeError, ValueError):
            pass
    out.setdefault("model_arch", arch_snapshot.arch.value)
    if arch_snapshot.is_moe:
        out.setdefault("expert_aware", True)
    return out


def compile_recipe(recipe: dict[str, Any], hw: HwDecision, profile: dict[str, Any]) -> BackendPlan:
    """Recipe + Profile + HwDecision → QuantIntent 与 export_format，组装 BackendPlan。"""
    model_id = recipe["model_id"]
    q = recipe.get("quant", {})
    arch_snapshot = resolve_model_arch(model_id, recipe=recipe)
    abstract_scheme, backend, wired_opts = resolve_quant_block(q, profile)
    ignore = merge_ignore_list(
        q.get("ignore"),
        arch_snapshot.arch,
        is_multimodal=arch_snapshot.is_multimodal,
    )
    backend_options = apply_arch_to_backend_options(
        backend,
        wired_opts,
        arch_snapshot,
    )
    if backend == "msmodelslim" and not backend_options.get("model_type"):
        backend_options.setdefault("model_type", Path(str(model_id)).name)
    calib = _scale_calib_for_arch(recipe.get("calib", {}), arch_snapshot)
    schemes_cfg = profile.get("schemes", {}) or {}
    scheme_cfg = schemes_cfg.get(abstract_scheme) or {}
    # 推理运行时由 profile.schemes.*.infer 决定，不要求 recipe 再写一遍
    infer_runtime = expected_infer_runtime(scheme_cfg) or "vllm_cuda"

    intent = QuantIntent(
        model_id=model_id,
        backend=backend,
        abstract_scheme=abstract_scheme,
        infer_runtime=infer_runtime,
        arch_snapshot=arch_snapshot,
        ignore=ignore,
        calib=calib,
        backend_options=backend_options,
    )

    compress = scheme_cfg.get("compress", {})
    export = compress.get("export")
    if export:
        try:
            export_format = ExportFormat(export)
        except ValueError:
            export_format = _BACKEND_EXPORT.get(
                intent.backend, ExportFormat.COMPRESSED_TENSORS
            )
    else:
        export_format = _SCHEME_EXPORT_OVERRIDE.get(
            intent.abstract_scheme,
            _BACKEND_EXPORT.get(intent.backend, ExportFormat.COMPRESSED_TENSORS),
        )

    if intent.abstract_scheme.startswith("ascend_") and intent.backend == "msmodelslim":
        export_format = ExportFormat.VLLM_ASCEND

    logger.info(
        "compile_recipe model_id=%s arch=%s backend=%s scheme=%s "
        "infer_runtime=%s export=%s profile=%s",
        model_id,
        arch_snapshot.arch.value,
        intent.backend,
        intent.abstract_scheme,
        intent.infer_runtime,
        export_format.value,
        hw.profile_id,
    )
    return BackendPlan(intent=intent, hw=hw, export_format=export_format)
