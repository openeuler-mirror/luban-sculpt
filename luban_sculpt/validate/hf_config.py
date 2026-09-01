"""Post-quant validation axis 1: HF config.json (+ optional luban manifest)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent

_CT_REQUIRED = ("config_groups", "format")
_VLLM_CT_METHOD = "compressed-tensors"

_METHOD_TO_EXPORT: dict[str, ExportFormat] = {
    "compressed-tensors": ExportFormat.COMPRESSED_TENSORS,
    "fp8": ExportFormat.FP8_HF,
    "awq": ExportFormat.AWQ_HF,
    "gptq": ExportFormat.GPTQ_HF,
    "ascend": ExportFormat.VLLM_ASCEND,
}


def plan_from_manifest(model_path: Path, manifest: dict[str, Any]) -> BackendPlan:
    """用 luban ``manifest.json`` 拼出静态校验用的 BackendPlan。"""
    export_str = manifest.get("export_format", "compressed-tensors")
    try:
        export_format = ExportFormat(export_str)
    except ValueError:
        export_format = ExportFormat.COMPRESSED_TENSORS
    return BackendPlan(
        intent=QuantIntent(
            model_id=str(model_path),
            backend=manifest.get("backend", "llm_compressor"),
            abstract_scheme=manifest.get("abstract_scheme", "fp8_dynamic"),
            deploy_target="vllm",
        ),
        hw=HwDecision(profile_id=manifest.get("profile_id", "generic_cpu")),
        export_format=export_format,
    )


def plan_from_hf_config(model_path: Path, config: dict[str, Any]) -> BackendPlan:
    """用标准 HF ``config.json`` / ``quantization_config`` 推断 BackendPlan。"""
    qc = config.get("quantization_config") or {}
    method = str(qc.get("quant_method") or "compressed-tensors").lower()
    export_format = _METHOD_TO_EXPORT.get(method, ExportFormat.COMPRESSED_TENSORS)
    backend = {
        ExportFormat.GPTQ_HF: "gptq",
        ExportFormat.AWQ_HF: "awq",
        ExportFormat.VLLM_ASCEND: "msmodelslim",
    }.get(export_format, "llm_compressor")
    return BackendPlan(
        intent=QuantIntent(
            model_id=str(model_path),
            backend=backend,
            abstract_scheme=str(qc.get("format") or method or "unknown"),
            deploy_target="vllm",
        ),
        hw=HwDecision(profile_id="unknown"),
        export_format=export_format,
    )


def validate_compressed_tensors_config(
    plan: BackendPlan, config: dict[str, Any]
) -> list[str]:
    """校验 HF ``config.json`` / ``quantization_config`` 是否满足 compressed-tensors 约定。"""
    errors: list[str] = []
    if plan.export_format != ExportFormat.COMPRESSED_TENSORS:
        return errors

    qc = config.get("quantization_config") or config
    method = qc.get("quant_method")
    if method and method != _VLLM_CT_METHOD:
        errors.append(f"quant_method expected {_VLLM_CT_METHOD!r}, got {method!r}")

    for key in _CT_REQUIRED:
        if key not in qc and key not in config:
            errors.append(f"missing compressed_tensors field: {key}")

    groups = qc.get("config_groups") or config.get("config_groups")
    if groups is not None and not isinstance(groups, (list, dict)):
        errors.append("config_groups must be list or dict")

    return errors


def validate_fp8_hf_config(config: dict[str, Any]) -> list[str]:
    qc = config.get("quantization_config") or config
    if qc.get("quant_method") != "fp8":
        return ["fp8_hf export expects quant_method=fp8"]
    return []


def validate_gptq_hf_config(config: dict[str, Any]) -> list[str]:
    qc = config.get("quantization_config") or config
    method = str(qc.get("quant_method") or "").lower()
    if method and method != "gptq":
        return [f"gptq export expects quant_method=gptq, got {method!r}"]
    if not method:
        return ["gptq export missing quantization_config.quant_method"]
    return []


def validate_awq_hf_config(config: dict[str, Any]) -> list[str]:
    qc = config.get("quantization_config") or config
    method = str(qc.get("quant_method") or "").lower()
    if method and method != "awq":
        return [f"awq export expects quant_method=awq, got {method!r}"]
    if not method:
        return ["awq export missing quantization_config.quant_method"]
    return []


def validate_hf_config(
    model_path: Path,
    *,
    plan: BackendPlan | None = None,
) -> list[str]:
    """量化产物 HF 配置校验，对齐标准 HF 量化目录。

    - **主文件**：``config.json``（社区 GPTQ/AWQ/CT 等均有）
    - **可选**：luban ``manifest.json``（仅 pipeline 产物；没有不报错）
    - **dry-run stub**：仅有 manifest、无 config → 放行（不跑字段检查）

    返回错误列表（空表示通过）。
    """
    if not model_path.is_dir():
        return [f"artifact dir missing: {model_path}"]

    config_path = model_path / "config.json"
    manifest_path = model_path / "manifest.json"
    has_config = config_path.is_file()
    has_manifest = manifest_path.is_file()

    if not has_config:
        if has_manifest:
            # luban dry-run / stub：只有 sidecar，尚无完整 HF 树
            return []
        return [f"missing config.json under {model_path}"]

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if plan is not None:
        active_plan = plan
    elif has_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        active_plan = plan_from_manifest(model_path, manifest)
    else:
        active_plan = plan_from_hf_config(model_path, config)

    errors = validate_compressed_tensors_config(active_plan, config)
    if active_plan.export_format == ExportFormat.FP8_HF:
        errors.extend(validate_fp8_hf_config(config))
    elif active_plan.export_format == ExportFormat.GPTQ_HF:
        errors.extend(validate_gptq_hf_config(config))
    elif active_plan.export_format == ExportFormat.AWQ_HF:
        errors.extend(validate_awq_hf_config(config))
    return errors
