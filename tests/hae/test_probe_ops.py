"""HAE ``probe_ops``：量化相关算子探测单测。"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from luban_sculpt.hae.probe_ops import (
    _matmul_probe,
    _pick_device,
    probe_one_op,
)


def test_matmul_probe_cpu_returns_latency_and_mse() -> None:
    """无加速卡时在 CPU 上跑 FP16 matmul，应给出有限延迟与非负 MSE。"""
    p50, mse = _matmul_probe("cpu")
    assert p50 >= 0.0
    assert mse >= 0.0


def test_probe_one_op_default_gemm_ok() -> None:
    """未在映射表中的算子回退到通用 matmul probe，默认应通过。"""
    result = probe_one_op(
        "cuda_fp8_gemm",
        vendor="unknown",
        timeout_sec=30.0,
        latency_threshold_ms=None,
        mse_threshold=0.01,
    )
    assert result["ok"] is True
    assert result["op"] == "cuda_fp8_gemm"
    assert result["latency_ms_p50"] is not None
    assert result["mse"] is not None
    assert "ok" in result["message"]


@pytest.mark.parametrize(
    "op",
    ["acl_int8_gemm", "npu_fused_matmul_scale", "acl_fp8_block"],
)
def test_probe_quant_ops_registered(op: str) -> None:
    """Profile 中常见的量化/融合 GEMM 类算子均可探测。"""
    result = probe_one_op(
        op,
        vendor="unknown",
        timeout_sec=30.0,
        latency_threshold_ms=None,
        mse_threshold=1.0,  # CPU FP16 宽松一点
    )
    assert result["ok"] is True
    assert result["latency_ms_p50"] is not None


def test_probe_forced_fail_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUBAN_PROBE_FAIL_OPS", "acl_int8_gemm,other_op")
    result = probe_one_op(
        "acl_int8_gemm",
        vendor="ascend",
        timeout_sec=5.0,
        latency_threshold_ms=None,
        mse_threshold=0.01,
    )
    assert result["ok"] is False
    assert "forced fail" in result["message"]
    assert result["latency_ms_p50"] is None


def test_probe_fails_when_mse_exceeds_threshold() -> None:
    with patch.dict(
        "luban_sculpt.hae.probe_ops._OP_RUNNERS",
        {"acl_fp8_block": lambda _d: (1.0, 0.5)},
    ):
        result = probe_one_op(
            "acl_fp8_block",
            vendor="ascend",
            timeout_sec=5.0,
            latency_threshold_ms=None,
            mse_threshold=0.01,
        )
    assert result["ok"] is False
    assert "mse=" in result["message"]


def test_probe_fails_when_latency_exceeds_threshold() -> None:
    with patch.dict(
        "luban_sculpt.hae.probe_ops._OP_RUNNERS",
        {"acl_int8_gemm": lambda _d: (100.0, 0.0)},
    ):
        result = probe_one_op(
            "acl_int8_gemm",
            vendor="ascend",
            timeout_sec=5.0,
            latency_threshold_ms=10.0,
            mse_threshold=0.01,
        )
    assert result["ok"] is False
    assert "p50=" in result["message"]


def test_probe_timeout() -> None:
    def _hang(_device: str) -> tuple[float, float]:
        import time

        time.sleep(2.0)
        return 0.0, 0.0

    with patch.dict("luban_sculpt.hae.probe_ops._OP_RUNNERS", {"slow_op": _hang}):
        result = probe_one_op(
            "slow_op",
            vendor="unknown",
            timeout_sec=0.1,
            latency_threshold_ms=None,
            mse_threshold=0.01,
        )
    assert result["ok"] is False
    assert "timeout" in result["message"]


def test_pick_device_falls_back_to_cpu() -> None:
    assert _pick_device("unknown") == "cpu"
