"""CalibRunner unit tests."""

from __future__ import annotations

from pathlib import Path

from luban_sculpt.calib import CalibRunner, default_perfectblend_dir
from luban_sculpt.contracts import BackendPlan, ExportFormat, HwDecision, QuantIntent


def _plan(**calib) -> BackendPlan:
    return BackendPlan(
        intent=QuantIntent(
            model_id="m",
            backend="gptq",
            abstract_scheme="w4_gptq",
            deploy_target="vllm",
            calib=dict(calib),
        ),
        hw=HwDecision(profile_id="generic_cpu"),
        export_format=ExportFormat.GPTQ_HF,
    )


def test_calib_runner_stub_load() -> None:
    data = CalibRunner(_plan(source="stub", max_samples=16)).load()
    assert data.num_samples == 8
    assert len(data.texts) == 8
    assert data.report["mode"] == "local"
    assert data.report["source"] == "stub"


def test_calib_runner_oneshot_kwargs_dataset_object() -> None:
    """有样本时 oneshot_kwargs 传物化 HF Dataset（text 列），不再传 hub id 字符串。"""
    runner = CalibRunner(_plan(source="stub", max_samples=64))
    data = runner.load()
    kw = runner.oneshot_kwargs(data)
    assert kw["num_calibration_samples"] == 64
    assert kw["text_column"] == "text"
    ds = kw["dataset"]
    assert hasattr(ds, "__len__")
    assert len(ds) == data.num_samples
    assert "text" in ds.column_names


def test_calib_runner_perfectblend_path() -> None:
    root = default_perfectblend_dir()
    if not root.exists():
        return
    runner = CalibRunner(
        _plan(
            source="open-perfectblend",
            path=str(root),
            max_samples=3,
            shuffle=True,
            seed=1,
        )
    )
    data = runner.load()
    assert data.num_samples == 3
    assert data.resolved_path == str(root.resolve())
    assert not data.texts[0].startswith("calib stub sample")
    kw = runner.oneshot_kwargs(data)
    assert len(kw["dataset"]) == 3


def test_calib_runner_write_jsonl(tmp_path: Path) -> None:
    runner = CalibRunner(_plan(source="stub", max_samples=8))
    data = runner.load()
    path = runner.write_jsonl(tmp_path / "luban_calib.jsonl", data)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == data.num_samples
    assert '"text"' in lines[0]
