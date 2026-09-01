"""华为 Ascend NPU 上量化算子探测测试（默认关闭）。

开启方式::

    LUBAN_TEST_NPU=1 pytest tests/test_hae_probe_ops_npu.py -q

可选::

    LUBAN_TEST_NPU_PROFILE=ascend_910b   # 默认 ascend_910b
"""

from __future__ import annotations

import os

import pytest

from luban_sculpt.hae.probe_ops import _matmul_probe, _pick_device, probe_one_op

_TRUE = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _npu_suite_enabled() -> bool:
    return _env_flag("LUBAN_TEST_NPU")


def _npu_ready() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return bool(getattr(torch, "npu", None) and torch.npu.is_available())
    except ImportError:
        return False


def _npu_profile_name() -> str:
    return os.environ.get("LUBAN_TEST_NPU_PROFILE", "ascend_910b").strip() or "ascend_910b"


pytestmark = pytest.mark.skipif(
    not _npu_suite_enabled(),
    reason="NPU probe tests disabled; set LUBAN_TEST_NPU=1 to enable",
)

requires_npu = pytest.mark.skipif(
    not _npu_ready(),
    reason="torch_npu is not available or no Ascend device",
)


@requires_npu
def test_npu_pick_device_ascend() -> None:
    assert _pick_device("ascend").startswith("npu")


@requires_npu
def test_npu_matmul_probe_on_device() -> None:
    p50, mse = _matmul_probe("npu:0")
    assert p50 >= 0.0
    assert mse >= 0.0
    assert p50 < 5000.0


@requires_npu
@pytest.mark.parametrize(
    "op",
    [
        "acl_int8_gemm",
        "npu_fused_matmul_scale",  # ascend_910b required_ops_probe
        "acl_fp8_block",
    ],
)
def test_npu_ascend_quant_ops_probe_one_op(op: str) -> None:
    """在 Ascend NPU 上探测华为侧量化算子名（当前 runner 仍为 FP16 matmul 占位）。"""
    result = probe_one_op(
        op,
        vendor="ascend",
        timeout_sec=120.0,
        latency_threshold_ms=None,
        mse_threshold=0.05,
    )
    assert result["ok"] is True, result["message"]
    assert result["latency_ms_p50"] is not None
    assert "npu" in result["message"]


@requires_npu
def test_npu_int8_weight_dequant_gemm() -> None:
    """简易 W8A16：INT8 权重反量化后与 FP16 激活做 GEMM（NPU）。"""
    import torch

    device = "npu:0"
    n, k = 128, 128
    w_i8 = torch.randint(-128, 127, (n, k), device=device, dtype=torch.int8)
    scale = torch.randn(n, device=device, dtype=torch.float16).abs() + 1e-3
    x = torch.randn(k, n, device=device, dtype=torch.float16)

    torch.npu.synchronize()
    w = w_i8.to(torch.float16) * scale[:, None]
    y = w @ x
    torch.npu.synchronize()

    assert y.shape == (n, n)
    assert torch.isfinite(y).all()

    ref = (w_i8.float() * scale[:, None].float()) @ x.float()
    mse = float(torch.mean((y.float() - ref) ** 2).item())
    assert mse < 1.0


@requires_npu
def test_npu_hae_probe_profile_required_ops() -> None:
    """端到端：按 Ascend profile 的 required_ops 在 NPU 上 probe。"""
    from luban_sculpt.hae.engine import HardwareAwareEngine, load_profile_template

    profile_name = _npu_profile_name()
    profile = load_profile_template(profile_name)
    ops = list(profile.get("required_ops_probe") or [])
    assert ops, f"{profile_name} should declare required_ops_probe"

    from luban_sculpt.hae.profile_fields import profile_soc_key

    hae = HardwareAwareEngine(
        profile_name,
        enable_real_probe=True,
        fallback_on_probe_failure=False,
        mse_threshold=0.05,
    )
    hae._identity = {
        "device_name": profile.get("display_name") or profile_soc_key(profile) or "Ascend",
        "vendor": "ascend",
        "stack": {"cann": os.environ.get("CANN_VERSION", "unknown")},
        "source": "test",
    }
    probe = hae.probe(profile)
    assert probe.ok is True, probe.messages
    assert probe.missing_ops == []
    assert any("npu" in m for m in probe.messages)
