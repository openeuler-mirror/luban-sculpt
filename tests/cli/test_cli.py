"""CLI 全部子命令模块测试（probe / compress / validate / report / backends / …）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from luban_sculpt.cli import (
    _cmd_ascend_chips,
    _cmd_backends,
    _cmd_compress,
    _cmd_model_arches,
    _cmd_probe,
    _cmd_report,
    _cmd_validate,
    main,
)
from luban_sculpt.contracts import HwDecision, ProbeResult
from tests.paths import RECIPES


@pytest.fixture(autouse=True)
def _clean_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUBAN_PROBE_FAIL_OPS", raising=False)
    monkeypatch.delenv("LUBAN_DEVICE_NAME", raising=False)
    monkeypatch.delenv("LUBAN_VENDOR", raising=False)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def test_cmd_probe_generic_cpu_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _cmd_probe(argparse.Namespace(profile="generic_cpu"))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["profile_id"] == "generic_cpu"
    assert out["probe_ok"] is True
    assert out["missing_ops"] == []
    assert out["hw_decision"]["profile_id"] == "generic_cpu"
    assert "fp8_dynamic" in out["profile_keys"]


def test_cmd_probe_ascend_910b_via_main(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["probe", "--profile", "ascend_910b"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["profile_id"] == "ascend_910b"
    assert out["vendor"] == "ascend"
    assert out["probe_ok"] is True
    assert "ascend_w8a8" in out["profile_keys"]
    assert out["hw_decision"]["comm_backend"] == "hccl"


def test_cmd_probe_failure_returns_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    hw = HwDecision(profile_id="ascend_910b", vendor="ascend")
    probe = ProbeResult(ok=False, messages=["fail"], missing_ops=["npu_fused_matmul_scale"])
    profile = {"schemes": {"ascend_w8a8": {}}}

    class _FakeHAE:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def run(self, *_a: Any, **_k: Any) -> tuple[HwDecision, ProbeResult, dict]:
            return hw, probe, profile

    with patch("luban_sculpt.cli.HardwareAwareEngine", _FakeHAE):
        rc = _cmd_probe(argparse.Namespace(profile="ascend_910b"))

    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["probe_ok"] is False
    assert out["missing_ops"] == ["npu_fused_matmul_scale"]


def test_cmd_probe_auto_profile_arg(capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}

    class _FakeHAE:
        def __init__(self, profile: str | None = None, **_k: Any) -> None:
            seen["ctor_profile"] = profile

        def run(self, profile_name: str | None = None) -> tuple[Any, Any, dict]:
            seen["run_profile"] = profile_name
            hw = HwDecision(profile_id="generic_cpu", vendor="unknown")
            probe = ProbeResult(ok=True, messages=[], missing_ops=[])
            return hw, probe, {"schemes": {}}

    with patch("luban_sculpt.cli.HardwareAwareEngine", _FakeHAE):
        rc = _cmd_probe(SimpleNamespace(profile="auto"))

    assert rc == 0
    assert seen["ctor_profile"] == "auto"
    assert seen["run_profile"] is None
    assert json.loads(capsys.readouterr().out)["probe_ok"] is True


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


def test_cmd_compress_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUBAN_LLM_COMPRESSOR_DRY_RUN", "1")
    recipe = RECIPES / "llama_fp8_dynamic.yaml"
    out = tmp_path / "compress_out"
    rc = _cmd_compress(
        argparse.Namespace(
            recipe=str(recipe),
            output=str(out),
            profile="generic_cpu",
        )
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["output"]).is_dir()
    assert (Path(payload["output"]) / "manifest.json").is_file()
    assert payload["manifest"]["profile_id"] == "generic_cpu"
    assert payload["manifest"]["backend"] == "llm_compressor"


def test_cmd_compress_via_main(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUBAN_LLM_COMPRESSOR_DRY_RUN", "1")
    recipe = RECIPES / "llama_fp8_dynamic.yaml"
    out = tmp_path / "main_out"
    rc = main(
        [
            "compress",
            "--recipe",
            str(recipe),
            "--output",
            str(out),
            "--profile",
            "generic_cpu",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "manifest" in payload


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_cmd_validate_ok_with_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text(
        json.dumps(
            {
                "profile_id": "generic_cpu",
                "export_format": "compressed-tensors",
                "backend": "llm_compressor",
                "abstract_scheme": "fp8_dynamic",
            }
        ),
        encoding="utf-8",
    )
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {},
                    "format": "float-quantized",
                }
            }
        ),
        encoding="utf-8",
    )

    with patch("luban_sculpt.cli.validate_hf_config", return_value=[]):
        rc = _cmd_validate(
            argparse.Namespace(
                model=str(model), runtime=False, runtime_mode="import"
            )
        )

    assert rc == 0
    text = capsys.readouterr().out
    assert "manifest profile_id=generic_cpu" in text
    assert "validate ok" in text


def test_cmd_validate_errors_return_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "bad"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")

    with patch(
        "luban_sculpt.cli.validate_hf_config",
        return_value=["missing quantization_config"],
    ):
        rc = _cmd_validate(
            argparse.Namespace(
                model=str(model), runtime=False, runtime_mode="import"
            )
        )

    assert rc == 1
    err = capsys.readouterr().err
    assert "missing quantization_config" in err


def test_cmd_validate_runtime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "gptq"}}),
        encoding="utf-8",
    )
    with patch("luban_sculpt.cli.validate_hf_config", return_value=[]):
        with patch(
            "luban_sculpt.cli.validate_runtime",
            return_value={"ok": True, "mode": "import"},
        ):
            rc = main(
                [
                    "validate",
                    "--model",
                    str(model),
                    "--runtime",
                    "--runtime-mode",
                    "import",
                ]
            )
    assert rc == 0
    assert "runtime:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_cmd_report_prints_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "m"
    model.mkdir()
    body = {"profile_id": "generic_cpu", "backend": "llm_compressor"}
    (model / "manifest.json").write_text(json.dumps(body), encoding="utf-8")

    rc = _cmd_report(argparse.Namespace(model=str(model)))
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == body


def test_cmd_report_missing_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "empty"
    model.mkdir()
    rc = main(["report", "--model", str(model)])
    assert rc == 1
    assert "no manifest.json" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# backends / ascend-chips / model-arches
# ---------------------------------------------------------------------------


def test_cmd_backends_lists_registry(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _cmd_backends(argparse.Namespace())
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    # entry-points 在 editable 安装下应能发现常见后端
    for name in ("llm_compressor", "msmodelslim", "awq", "gptq"):
        if name in data:
            break
    else:
        # 至少是合法 JSON 列表（未安装 entry-points 时可能为空）
        assert data == [] or all(isinstance(x, str) for x in data)


def test_cmd_backends_via_main(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["backends"]) == 0
    json.loads(capsys.readouterr().out)


def test_cmd_ascend_chips_prints_rows(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _cmd_ascend_chips(argparse.Namespace())
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert isinstance(rows, list)
    names = {r["profile_name"] for r in rows}
    assert names == {"ascend_910b"}
    sample = rows[0]
    assert sample["fp8_native"] is False
    assert "w8a8" in sample["allowed_quant_types"]
    assert "fp8" not in sample["allowed_quant_types"]


def test_cmd_ascend_chips_via_main(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ascend-chips"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_cmd_model_arches_prints_list(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _cmd_model_arches(argparse.Namespace())
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert isinstance(rows, list)
    assert len(rows) >= 1


def test_cmd_model_arches_via_main(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model-arches"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)
