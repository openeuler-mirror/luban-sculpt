"""msmodelslim backend + CalibRunner wiring."""

from __future__ import annotations

from pathlib import Path

from luban_sculpt.backends.msmodelslim.runner import (
    build_quant_argv,
    resolve_device,
    run_msmodelslim_quant,
)
from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent


def _plan(**opts) -> BackendPlan:
    return BackendPlan(
        intent=QuantIntent(
            model_id="/data/models/Qwen2.5-7B-Instruct",
            backend="msmodelslim",
            abstract_scheme="ascend_w8a8",
            deploy_target="vllm_ascend",
            calib={"source": "stub", "max_samples": 8},
            backend_options={
                "model_type": "Qwen2.5-7B-Instruct",
                "quant_type": "w8a8",
                **opts,
            },
        ),
        hw=HwDecision(profile_id="ascend_910b"),
        export_format=ExportFormat.COMPRESSED_TENSORS,
    )


def test_msmodelslim_uses_calib_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUBAN_MSMODELSLIM_DRY_RUN", "1")
    meta = run_msmodelslim_quant(_plan(), tmp_path)
    assert meta["status"] == "dry_run"
    assert meta["calib_report"]["source"] == "stub"
    assert (tmp_path / "luban_calib.jsonl").is_file()
    assert "--calib_file" not in meta["cli"]
    assert "--device_id" not in meta["cli"]
    assert meta["cli"][meta["cli"].index("--device") + 1] == "npu"


def test_resolve_device_formats() -> None:
    assert resolve_device({}) == "npu"
    assert resolve_device({"device": "npu:0,1"}) == "npu:0,1"
    assert resolve_device({"device": "npu", "device_id": [0, 1, 2]}) == "npu:0,1,2"
    assert resolve_device({"device": "npu", "device_id": 0}) == "npu:0"


def test_build_quant_argv_config_path_and_tag() -> None:
    argv = build_quant_argv(
        _plan(config_path="/tmp/q.yaml", tag=["vllm"], debug=True, device="npu:0"),
        Path("/tmp/out"),
    )
    assert "--config_path" in argv and "/tmp/q.yaml" in argv
    assert "--quant_type" not in argv
    assert argv[argv.index("--device") + 1] == "npu:0"
    assert "--debug" in argv
    assert argv[argv.index("--tag") + 1] == "vllm"


def test_run_streaming_tees_logs(tmp_path, caplog) -> None:
    import logging
    import sys

    from luban_sculpt.backends.msmodelslim.runner import _run_streaming

    out = tmp_path / "out.log"
    err = tmp_path / "err.log"
    with caplog.at_level(logging.INFO):
        rc = _run_streaming(
            [
                sys.executable,
                "-c",
                "import sys; print('hello-ms'); print('warn-ms', file=sys.stderr)",
            ],
            stdout_path=out,
            stderr_path=err,
        )
    assert rc == 0
    assert "hello-ms" in out.read_text(encoding="utf-8")
    assert "warn-ms" in err.read_text(encoding="utf-8")
    assert any("hello-ms" in r.message for r in caplog.records)
