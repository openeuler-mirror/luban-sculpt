"""Invoke MindStudio msModelSlim CLI (``msmodelslim quant``).

Reference: https://github.com/Ascend/msmodelslim
Quick start: ``msmodelslim quant --model_path ... --save_path ... --quant_type w8a8``
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from luban_sculpt.backends.msmodelslim.chips import (
    assert_quant_type_allowed,
    get_chip,
    normalize_soc_key,
)
from luban_sculpt.calib import CalibRunner
from luban_sculpt.contracts import BackendPlan
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


class MsModelSlimNotInstalledError(RuntimeError):
    pass


def msmodelslim_cli() -> str:
    return os.environ.get("MSMODELSLIM_BIN", "msmodelslim")


def is_msmodelslim_available() -> bool:
    return shutil.which(msmodelslim_cli()) is not None


def resolve_quant_type(plan: BackendPlan) -> str:
    opts = plan.intent.backend_options or {}
    qt = opts.get("quant_type")
    if qt:
        return str(qt).lower()
    scheme = plan.intent.abstract_scheme
    if scheme in ("ascend_fp8", "fp8_native"):
        return "fp8"
    if scheme in ("ascend_w4a8", "w4a8"):
        return "w4a8"
    return "w8a8"


def resolve_soc_key(plan: BackendPlan) -> str:
    opts = plan.intent.backend_options or {}
    if opts.get("ascend_soc"):
        key = normalize_soc_key(str(opts["ascend_soc"]))
        if key:
            return key
    env = os.environ.get("LUBAN_ASCEND_SOC") or os.environ.get("ASCEND_SOC_VERSION")
    if env:
        key = normalize_soc_key(env)
        if key:
            return key
    pid = plan.hw.profile_id
    for suffix in ("910b",):
        if suffix in pid:
            return suffix
    return "910b"


def build_quant_argv(
    plan: BackendPlan,
    output_dir: Path,
    *,
    calib_file: Path | None = None,
) -> list[str]:
    opts = plan.intent.backend_options or {}
    quant_type = resolve_quant_type(plan)
    chip = get_chip(resolve_soc_key(plan))
    assert_quant_type_allowed(chip, quant_type)

    model_path = opts.get("model_path") or plan.intent.model_id
    save_path = opts.get("save_path") or str(output_dir)
    model_type = opts.get("model_type")
    if not model_type:
        # Derive lab-style name from path basename; still required by msModelSlim CLI
        model_type = Path(str(model_path)).name or None
    if not model_type:
        raise ValueError(
            "msmodelslim backend requires quant.msmodelslim.model_type "
            "(e.g. Qwen2.5-7B-Instruct) — see msModelSlim lab_practice / docs; "
            f"model_arch={plan.intent.arch_snapshot.arch.value}"
        )
    logger.info(
        "msmodelslim model_type=%s model_arch=%s",
        model_type,
        plan.intent.arch_snapshot.arch.value,
    )

    device = opts.get("device", "npu")
    trust = opts.get("trust_remote_code", True)

    argv = [
        msmodelslim_cli(),
        "quant",
        "--model_path",
        str(model_path),
        "--save_path",
        str(save_path),
        "--device",
        str(device),
        "--model_type",
        str(model_type),
        "--quant_type",
        quant_type,
        "--trust_remote_code",
        "True" if trust else "False",
    ]

    device_ids = opts.get("device_id")
    if device_ids is not None:
        if isinstance(device_ids, (list, tuple)):
            argv.extend(["--device_id", *[str(x) for x in device_ids]])
        else:
            argv.extend(["--device_id", str(device_ids)])

    # V1 一键量化默认用 lab_calib；显式 calib_file / pass_calib_cli 时注入（V0/forks）
    pass_cli = bool(opts.get("pass_calib_cli") or opts.get("calib_file"))
    if pass_cli and calib_file is not None:
        argv.extend(["--calib_file", str(calib_file)])

    extra_args = opts.get("extra_cli") or []
    if isinstance(extra_args, list):
        argv.extend([str(x) for x in extra_args])

    return argv


def run_msmodelslim_quant(
    plan: BackendPlan,
    output_dir: Path,
    *,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """调用 ``msmodelslim quant``；CLI 缺失或 LUBAN_MSMODELSLIM_DRY_RUN 时只写 argv 元数据。

    校准经共享 ``CalibRunner``：写入 ``calib_report``，并物化 ``luban_calib.jsonl``。
    V1 CLI 默认内置 lab 校准；设 ``quant.msmodelslim.calib_file`` 或
    ``pass_calib_cli: true`` 时再把路径传给 ``--calib_file``。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = plan.intent.backend_options or {}
    calib_runner = CalibRunner(plan)
    calib_data = calib_runner.load()
    calib_file = calib_runner.resolve_calib_file(
        output_dir,
        calib_data,
        explicit=opts.get("calib_file"),
    )
    argv = build_quant_argv(plan, output_dir, calib_file=calib_file)
    chip = get_chip(resolve_soc_key(plan))
    meta = {
        "cli": argv,
        "ascend_soc": chip.soc,
        "quant_type": resolve_quant_type(plan),
        "export": "vllm_ascend",
        "vllm_serve_hint": (
            f"vllm serve {output_dir} --quantization {chip.vllm_quantization}"
        ),
        "calib_report": calib_data.report,
    }
    if calib_file is not None:
        meta["calib_file"] = str(calib_file)

    if dry_run is None:
        dry_run = os.environ.get("LUBAN_MSMODELSLIM_DRY_RUN", "").lower() in (
            "1",
            "true",
            "yes",
        )

    if dry_run or not is_msmodelslim_available():
        if not dry_run and not is_msmodelslim_available():
            logger.warning(
                "%s not in PATH; writing plan only. Install msModelSlim or set "
                "LUBAN_MSMODELSLIM_DRY_RUN=1. See example/msmodelslim/",
                msmodelslim_cli(),
            )
        logger.info(
            "msmodelslim dry-run quant_type=%s output=%s",
            meta.get("quant_type"),
            output_dir,
        )
        (output_dir / "msmodelslim_command.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            + " ".join(_shell_quote(a) for a in argv)
            + "\n",
            encoding="utf-8",
        )
        meta["status"] = "dry_run"
        return meta

    logger.info("Running: %s", " ".join(argv))
    proc = subprocess.run(argv, capture_output=True, text=True)
    meta["status"] = "ok" if proc.returncode == 0 else "failed"
    meta["returncode"] = proc.returncode
    if proc.stdout:
        (output_dir / "msmodelslim_stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.stderr:
        (output_dir / "msmodelslim_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"msmodelslim quant failed (code {proc.returncode}); "
            f"see {output_dir}/msmodelslim_stderr.log"
        )
    return meta


def _shell_quote(s: str) -> str:
    if not s:
        return "''"
    if all(c.isalnum() or c in "/._-:" for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"
