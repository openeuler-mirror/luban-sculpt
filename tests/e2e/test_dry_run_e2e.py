"""Dry-run 完整链路：probe → compile → compress → validate → report。

覆盖 SDK（QuantPipeline）与 CLI，在无加速卡 / 无量化工具时仍可跑通。
单阶段 recipe 产物落在 ``<output>/stage_0_<backend>/``；根目录另有 ``pipeline_manifest.json``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from luban_sculpt.cli import (
    _cmd_backends,
    _cmd_compress,
    _cmd_probe,
    _cmd_report,
    _cmd_validate,
    main,
)
from luban_sculpt.compiler.plan import compile_plan
from luban_sculpt.contracts import ExportFormat
from luban_sculpt.hae.engine import HardwareAwareEngine
from luban_sculpt.pipeline import QuantPipeline
from tests.paths import RECIPES

PROFILE = "generic_cpu"


@pytest.fixture(autouse=True)
def _dry_run_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUBAN_LLM_COMPRESSOR_DRY_RUN", "1")
    monkeypatch.setenv("LUBAN_GPTQMODEL_DRY_RUN", "1")
    monkeypatch.setenv("LUBAN_MSMODELSLIM_DRY_RUN", "1")
    monkeypatch.delenv("LUBAN_PROBE_FAIL_OPS", raising=False)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 单阶段：HAE → compile → pipeline(+validate) → CLI validate/report
# ---------------------------------------------------------------------------


def test_e2e_dry_run_single_stage_full_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = RECIPES / "llama_fp8_dynamic.yaml"
    out = tmp_path / "e2e_single"

    # 1) HAE
    hw, probe, profile = HardwareAwareEngine(PROFILE).run(PROFILE)
    assert hw.profile_id == PROFILE
    assert probe.ok is True
    assert "fp8_dynamic" in profile.get("schemes", {})

    # 2) compile
    plan = compile_plan(recipe, PROFILE)
    assert plan.intent.backend == "llm_compressor"
    assert plan.intent.abstract_scheme == "fp8_dynamic"
    assert plan.intent.calib.get("source") == "stub"
    assert plan.export_format == ExportFormat.COMPRESSED_TENSORS
    assert plan.hw.profile_id == PROFILE

    # 3) compress + 压后 HF 校验（dry-run 仅有 manifest 亦可）
    artifact = QuantPipeline(
        profile_name=PROFILE,
        validate_quantized_model=True,
    ).run(recipe, out)

    stage_out = artifact.output_dir
    assert stage_out == out / "stage_0_llm_compressor"
    assert (out / "pipeline_manifest.json").is_file()
    pipe = _read_json(out / "pipeline_manifest.json")
    assert pipe["profile_id"] == PROFILE
    assert pipe["stages"][0]["backend"] == "llm_compressor"

    manifest = _read_json(stage_out / "manifest.json")
    assert manifest["profile_id"] == PROFILE
    assert manifest["backend"] == "llm_compressor"
    assert manifest["abstract_scheme"] == "fp8_dynamic"
    assert manifest["export_format"] == "compressed-tensors"
    assert "vllm_launch" in manifest

    assert (stage_out / "llm_compressor_oneshot.py").is_file()
    assert (stage_out / "recipe_stub.json").is_file()
    assert (stage_out / "quant_stub.json").is_file()
    assert (stage_out / "pipeline_stage.json").is_file()

    vr = _read_json(stage_out / "quantized_model_validate.json")
    assert vr["ok"] is True
    assert vr["check_hf_config"] is True

    # 4) CLI validate（指向阶段产物目录）
    rc = _cmd_validate(
        argparse.Namespace(model=str(stage_out), runtime=False, runtime_mode="import")
    )
    assert rc == 0
    assert "validate ok" in capsys.readouterr().out

    # 5) CLI report
    rc = _cmd_report(argparse.Namespace(model=str(stage_out)))
    assert rc == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["backend"] == "llm_compressor"
    assert reported["profile_id"] == PROFILE


def test_e2e_dry_run_cli_probe_compress_validate_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI 完整子命令链（与用户操作路径一致）。"""
    recipe = RECIPES / "llama_fp8_dynamic.yaml"
    out = tmp_path / "cli_chain"

    rc = _cmd_probe(argparse.Namespace(profile=PROFILE))
    assert rc == 0
    probe_out = json.loads(capsys.readouterr().out)
    assert probe_out["profile_id"] == PROFILE
    assert probe_out["probe_ok"] is True
    assert "fp8_dynamic" in probe_out["profile_keys"]

    rc = _cmd_compress(
        argparse.Namespace(
            recipe=str(recipe),
            output=str(out),
            profile=PROFILE,
            validate_quantized_model=True,
            validate_runtime=False,
            runtime_mode="import",
        )
    )
    assert rc == 0
    compress_payload = json.loads(capsys.readouterr().out)
    stage_out = Path(compress_payload["output"])
    assert stage_out == out / "stage_0_llm_compressor"
    assert compress_payload["manifest"]["backend"] == "llm_compressor"
    assert (stage_out / "manifest.json").is_file()
    assert (stage_out / "quantized_model_validate.json").is_file()
    assert (stage_out / "llm_compressor_oneshot.py").is_file()
    assert (out / "pipeline_manifest.json").is_file()

    rc = _cmd_validate(
        argparse.Namespace(model=str(stage_out), runtime=False, runtime_mode="import")
    )
    assert rc == 0
    assert "validate ok" in capsys.readouterr().out

    rc = _cmd_report(argparse.Namespace(model=str(stage_out)))
    assert rc == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["profile_id"] == PROFILE

    rc = _cmd_backends(argparse.Namespace())
    assert rc == 0
    names = json.loads(capsys.readouterr().out)
    assert "llm_compressor" in names
    assert "gptq" in names
    assert "msmodelslim" in names

    rc = main(["report", "--model", str(stage_out)])
    assert rc == 0


# ---------------------------------------------------------------------------
# 多阶段 dry-run
# ---------------------------------------------------------------------------


def test_e2e_dry_run_multi_stage_pipeline(tmp_path: Path) -> None:
    recipe = RECIPES / "pipeline_llm_compressor_then_gptq.yaml"
    out = tmp_path / "e2e_multi"

    artifact = QuantPipeline(
        profile_name=PROFILE,
        validate_quantized_model=True,
    ).run(recipe, out)

    pipe_manifest = _read_json(out / "pipeline_manifest.json")
    assert pipe_manifest["profile_id"] == PROFILE
    assert len(pipe_manifest["stages"]) == 2
    assert pipe_manifest["stages"][0]["backend"] == "llm_compressor"
    assert pipe_manifest["stages"][1]["backend"] == "gptq"
    assert pipe_manifest["final_output"].endswith("stage2_gptq")

    stage1 = out / "stage1_fp8"
    stage2 = out / "stage2_gptq"
    assert _read_json(stage1 / "manifest.json")["backend"] == "llm_compressor"
    assert _read_json(stage2 / "manifest.json")["backend"] == "gptq"
    assert _read_json(stage2 / "manifest.json")["export_format"] == "gptq_hf"

    assert (stage1 / "llm_compressor_oneshot.py").is_file()
    assert (stage1 / "quant_stub.json").is_file()
    assert (stage1 / "pipeline_stage.json").is_file()
    assert (stage2 / "quant_stub.json").is_file()
    assert (stage2 / "pipeline_stage.json").is_file()

    assert artifact.output_dir == stage2
    assert _read_json(stage2 / "quantized_model_validate.json")["ok"] is True


# ---------------------------------------------------------------------------
# Ascend / msmodelslim dry-run
# ---------------------------------------------------------------------------


def test_e2e_dry_run_ascend_msmodelslim(tmp_path: Path) -> None:
    recipe = RECIPES / "ascend_qwen_w8a8.yaml"
    out = tmp_path / "e2e_ascend"
    profile = "ascend_910b"

    plan = compile_plan(recipe, profile)
    assert plan.intent.backend == "msmodelslim"
    assert plan.intent.abstract_scheme == "ascend_w8a8"
    assert plan.export_format == ExportFormat.VLLM_ASCEND

    artifact = QuantPipeline(profile_name=profile).run(recipe, out)
    stage_out = artifact.output_dir
    assert stage_out == out / "stage_0_msmodelslim"

    manifest = _read_json(stage_out / "manifest.json")
    assert manifest["backend"] == "msmodelslim"
    assert manifest["export_format"] == "vllm_ascend"
    assert manifest["profile_id"] == profile

    assert (stage_out / "msmodelslim_command.sh").is_file()
    assert (stage_out / "luban_calib.jsonl").is_file()
    assert (stage_out / "quant_stub.json").is_file()
    assert (out / "pipeline_manifest.json").is_file()

    cmd = (stage_out / "msmodelslim_command.sh").read_text(encoding="utf-8")
    assert "msmodelslim" in cmd and "quant" in cmd
    assert "--quant_type" in cmd

    calib_lines = (
        (stage_out / "luban_calib.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(calib_lines) >= 1
    assert "text" in json.loads(calib_lines[0])
