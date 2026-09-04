"""llm_compressor runner helpers（不依赖真实 oneshot）。"""

from __future__ import annotations

from luban_sculpt.backends.llm_compressor.runner import _maybe_sequential_pipeline


def _fake_oneshot_with_pipeline(*, pipeline: str = "independent"):
    def oneshot(*, model=None, recipe=None, pipeline=pipeline, **kwargs):
        return None

    return oneshot


def _fake_oneshot_no_pipeline():
    def oneshot(*, model=None, recipe=None, **kwargs):
        return None

    return oneshot


def test_pipeline_explicit_sequential() -> None:
    out = _maybe_sequential_pipeline(
        _fake_oneshot_with_pipeline(),
        {},
        {"pipeline": "sequential"},
        is_gptq=False,
    )
    assert out["pipeline"] == "sequential"


def test_pipeline_from_oneshot_kwargs() -> None:
    out = _maybe_sequential_pipeline(
        _fake_oneshot_with_pipeline(),
        {"pipeline": "sequential"},
        {},
        is_gptq=False,
    )
    assert out["pipeline"] == "sequential"


def test_gptq_defaults_to_sequential() -> None:
    out = _maybe_sequential_pipeline(
        _fake_oneshot_with_pipeline(),
        {},
        {},
        is_gptq=True,
    )
    assert out["pipeline"] == "sequential"


def test_pipeline_false_disables_gptq_default() -> None:
    out = _maybe_sequential_pipeline(
        _fake_oneshot_with_pipeline(),
        {"pipeline": False},
        {},
        is_gptq=True,
    )
    assert "pipeline" not in out


def test_skip_when_oneshot_lacks_pipeline_param() -> None:
    out = _maybe_sequential_pipeline(
        _fake_oneshot_no_pipeline(),
        {},
        {"pipeline": "sequential"},
        is_gptq=True,
    )
    assert "pipeline" not in out
