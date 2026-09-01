"""Orchestrate quantized-model checks: HF config then optional vLLM runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from luban_sculpt.contracts import BackendPlan
from luban_sculpt.validate.hf_config import validate_hf_config
from luban_sculpt.validate.runtime import RuntimeMode, validate_runtime


class QuantizedModelValidateError(RuntimeError):
    """量化模型目录校验失败（HF 配置或运行时）。"""


def validate_quantized_model(
    model_path: Path,
    *,
    plan: BackendPlan | None = None,
    check_hf_config: bool = True,
    check_runtime: bool = False,
    runtime_mode: RuntimeMode = "import",
) -> dict[str, Any]:
    """对量化模型目录做统一校验（编排 ``hf_config`` + ``runtime``）。

    1. **HF 配置**：``config.json``（可选 luban ``manifest.json``）
    2. **运行时**：可选 ``import`` / ``load`` / ``generate``

    返回报告 dict（含 ``ok`` / ``hf_config_errors`` / ``runtime``）；不抛异常。
    """
    hf_config_errors: list[str] = []
    if check_hf_config:
        hf_config_errors = validate_hf_config(model_path, plan=plan)

    runtime_result: dict[str, Any] | None = None
    if check_runtime:
        runtime_result = validate_runtime(model_path, mode=runtime_mode)

    runtime_ok = True if runtime_result is None else bool(runtime_result.get("ok"))
    ok = not hf_config_errors and runtime_ok
    report: dict[str, Any] = {
        "ok": ok,
        "path": str(model_path),
        "check_hf_config": check_hf_config,
        "check_runtime": check_runtime,
        "hf_config_errors": hf_config_errors,
        "runtime": runtime_result,
    }
    if not ok:
        errs = list(hf_config_errors)
        if runtime_result and not runtime_result.get("ok"):
            errs.append(
                runtime_result.get("error")
                or runtime_result.get("stderr")
                or "runtime check failed"
            )
        report["errors"] = errs
    return report


def write_quantized_model_report(model_path: Path, report: dict[str, Any]) -> Path:
    path = model_path / "quantized_model_validate.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
