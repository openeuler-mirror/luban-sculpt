"""YAML Recipe → QuantIntent + BackendPlan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from luban_sculpt.model import (
    apply_arch_to_backend_options,
    resolve_model_arch,
    merge_ignore_list,
)
from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent
from luban_sculpt.log import get_logger

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
    "fp8_dynamic": ExportFormat.COMPRESSED_TENSORS,
    "fp8_block": ExportFormat.COMPRESSED_TENSORS,
    "fp8_hf": ExportFormat.FP8_HF,
    "ascend_w8a8": ExportFormat.VLLM_ASCEND,
    "ascend_w4a8": ExportFormat.VLLM_ASCEND,
    "ascend_fp8": ExportFormat.VLLM_ASCEND,
    "ascend_w8a16": ExportFormat.VLLM_ASCEND,
    "w4_gptq": ExportFormat.GPTQ_HF,
    "w4a16_awq": ExportFormat.AWQ_HF,
    "w4a16": ExportFormat.COMPRESSED_TENSORS,
    "w4a16_g32": ExportFormat.COMPRESSED_TENSORS,
    "hygon_w4a16_awq": ExportFormat.AWQ_HF,
    "hygon_w8a8_gptq": ExportFormat.GPTQ_HF,
    "slimquant_w4a8": ExportFormat.COMPRESSED_TENSORS,
    "slimquant_w4a8_marlin": ExportFormat.COMPRESSED_TENSORS,
    "blockwise_int8": ExportFormat.COMPRESSED_TENSORS,
    "awq_marlin": ExportFormat.COMPRESSED_TENSORS,
}


def load_recipe_yaml(path: Path) -> dict[str, Any]:
    """读取 recipe YAML（model_id / quant / calib）。"""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    backend = q.get("backend", "llm_compressor")
    arch_snapshot = resolve_model_arch(model_id, recipe=recipe)
    ignore = merge_ignore_list(
        q.get("ignore"),
        arch_snapshot.arch,
        is_multimodal=arch_snapshot.is_multimodal,
    )
    backend_options = apply_arch_to_backend_options(
        backend,
        q.get(backend, q.get("backend_options", {})),
        arch_snapshot,
    )
    calib = _scale_calib_for_arch(recipe.get("calib", {}), arch_snapshot)

    intent = QuantIntent(
        model_id=model_id,
        backend=backend,
        abstract_scheme=q.get("abstract_scheme", q.get("scheme", "fp8_dynamic")),
        deploy_target=q.get("deploy_target", "vllm_cuda"),
        arch_snapshot=arch_snapshot,
        ignore=ignore,
        calib=calib,
        backend_options=backend_options,
    )

    scheme_cfg = profile.get("schemes", {}).get(intent.abstract_scheme, {})
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
        "compile_recipe model_id=%s arch=%s backend=%s scheme=%s export=%s profile=%s",
        model_id,
        arch_snapshot.arch.value,
        intent.backend,
        intent.abstract_scheme,
        export_format.value,
        hw.profile_id,
    )
    return BackendPlan(intent=intent, hw=hw, export_format=export_format)
