"""Invoke MindStudio msModelSlim CLI (``msmodelslim quant``).

Reference: https://github.com/Ascend/msmodelslim

CLI (current):
  msmodelslim quant --model_path ... --save_path ...
    [--model_type ...] [--device npu|npu:0,1,...] [--config_path ...]
    [--quant_type w8a8] [--trust_remote_code True|False] [--debug] [--tag ...]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
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


def resolve_device(opts: dict[str, Any]) -> str:
    """Build ``--device`` value: ``npu`` / ``cpu`` / ``npu:0,1,2,3``.

    Recipe may set ``device: npu:0,1`` directly, or ``device: npu`` + ``device_id: [0, 1]``.
    """
    device = str(opts.get("device", "npu")).strip() or "npu"
    if ":" in device:
        return device
    device_ids = opts.get("device_id")
    if device_ids is None:
        return device
    if isinstance(device_ids, (list, tuple)):
        ids = ",".join(str(x) for x in device_ids)
    else:
        ids = str(device_ids).replace(" ", ",")
    if not ids:
        return device
    return f"{device}:{ids}"


def build_quant_argv(
    plan: BackendPlan,
    output_dir: Path,
    *,
    calib_file: Path | None = None,
) -> list[str]:
    del calib_file  # CalibRunner still materializes jsonl; V1 CLI has no --calib_file
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

    device = resolve_device(opts)
    trust = opts.get("trust_remote_code", True)
    config_path = opts.get("config_path") or opts.get("config")

    argv = [
        msmodelslim_cli(),
        "quant",
        "--model_path",
        str(model_path),
        "--save_path",
        str(save_path),
        "--device",
        device,
        "--model_type",
        str(model_type),
        "--trust_remote_code",
        "True" if trust else "False",
    ]

    # --config_path 与 --quant_type 互斥；有显式配置时不再传 quant_type
    if config_path:
        argv.extend(["--config_path", str(config_path)])
    else:
        argv.extend(["--quant_type", quant_type])

    if opts.get("debug"):
        argv.append("--debug")

    tags = opts.get("tag") or opts.get("tags")
    if tags:
        if isinstance(tags, str):
            tag_list = [tags]
        else:
            tag_list = [str(t) for t in tags]
        if tag_list:
            argv.extend(["--tag", *tag_list])

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
    当前 msModelSlim V1 CLI 无 ``--calib_file``，校准由 lab_practice / ``--config_path`` 配置。
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
    stdout_path = output_dir / "msmodelslim_stdout.log"
    stderr_path = output_dir / "msmodelslim_stderr.log"
    returncode = _run_streaming(argv, stdout_path=stdout_path, stderr_path=stderr_path)
    meta["status"] = "ok" if returncode == 0 else "failed"
    meta["returncode"] = returncode
    meta["stdout_log"] = str(stdout_path)
    meta["stderr_log"] = str(stderr_path)
    if returncode != 0:
        raise RuntimeError(
            f"msmodelslim quant failed (code {returncode}); "
            f"see {stderr_path}"
        )
    return meta


def _run_streaming(
    argv: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    """Run CLI; tee stdout/stderr line-by-line to logger and log files."""
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None and proc.stderr is not None

    def _tee(pipe, path: Path, *, is_stderr: bool) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for line in pipe:
                fh.write(line)
                fh.flush()
                msg = line.rstrip("\n")
                if is_stderr:
                    logger.warning("[msmodelslim:stderr] %s", msg)
                else:
                    logger.info("[msmodelslim] %s", msg)

    threads = [
        threading.Thread(
            target=_tee, args=(proc.stdout, stdout_path), kwargs={"is_stderr": False}, daemon=True
        ),
        threading.Thread(
            target=_tee, args=(proc.stderr, stderr_path), kwargs={"is_stderr": True}, daemon=True
        ),
    ]
    for t in threads:
        t.start()
    returncode = proc.wait()
    for t in threads:
        t.join()
    return int(returncode)


def _shell_quote(s: str) -> str:
    if not s:
        return "''"
    if all(c.isalnum() or c in "/._-:" for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"
