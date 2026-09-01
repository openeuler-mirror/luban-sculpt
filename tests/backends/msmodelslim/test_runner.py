"""msmodelslim backend + CalibRunner wiring."""

from __future__ import annotations

from luban_sculpt.backends.msmodelslim.runner import run_msmodelslim_quant
from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent


def test_msmodelslim_uses_calib_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUBAN_MSMODELSLIM_DRY_RUN", "1")
    plan = BackendPlan(
        intent=QuantIntent(
            model_id="/data/models/Qwen2.5-7B-Instruct",
            backend="msmodelslim",
            abstract_scheme="ascend_w8a8",
            deploy_target="vllm_ascend",
            calib={"source": "stub", "max_samples": 8},
            backend_options={
                "model_type": "Qwen2.5-7B-Instruct",
                "quant_type": "w8a8",
            },
        ),
        hw=HwDecision(profile_id="ascend_910b"),
        export_format=ExportFormat.COMPRESSED_TENSORS,
    )
    meta = run_msmodelslim_quant(plan, tmp_path)
    assert meta["status"] == "dry_run"
    assert meta["calib_report"]["source"] == "stub"
    assert (tmp_path / "luban_calib.jsonl").is_file()
    assert "--calib_file" not in meta["cli"]
