"""GPU 上量化相关算子探测测试（默认关闭）。

开启方式::

    LUBAN_TEST_GPU=1 pytest tests/test_hae_probe_ops_gpu.py -q

可选::

    LUBAN_TEST_GPU_VENDOR=nvidia   # 默认 nvidia；亦可 hygon（ROCm cuda 设备）
"""

from __future__ import annotations

import os

import pytest

from luban_sculpt.hae.probe_ops import _matmul_probe, _pick_device, probe_one_op

_TRUE = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _gpu_suite_enabled() -> bool:
    return _env_flag("LUBAN_TEST_GPU")


def _cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _gpu_vendor() -> str:
    return os.environ.get("LUBAN_TEST_GPU_VENDOR", "nvidia").strip().lower() or "nvidia"


pytestmark = pytest.mark.skipif(
    not _gpu_suite_enabled(),
    reason="GPU probe tests disabled; set LUBAN_TEST_GPU=1 to enable",
)

requires_cuda = pytest.mark.skipif(
    not _cuda_ready(),
    reason="torch.cuda is not available",
)


@requires_cuda
def test_gpu_pick_device_nvidia_or_hygon() -> None:
    vendor = _gpu_vendor()
    if vendor not in ("nvidia", "hygon"):
        pytest.skip(f"unsupported LUBAN_TEST_GPU_VENDOR={vendor}")
    assert _pick_device(vendor).startswith("cuda")


@requires_cuda
def test_gpu_matmul_probe_on_cuda() -> None:
    p50, mse = _matmul_probe("cuda:0")
    assert p50 >= 0.0
    assert mse >= 0.0
    # GPU 小矩阵通常应明显快于「假超时」量级
    assert p50 < 500.0


@requires_cuda
@pytest.mark.parametrize(
    "op",
    [
        "cuda_fp8_gemm",  # nvidia_h20 / h100 required_ops_probe
        "cutlass_fp8",  # nvidia_h20 scheme kernel_whitelist
        "scaled_mm_fp8",
        "cuda_int8_gemm",  # W8 路径占位（FP16 matmul 烟测）
    ],
)
def test_gpu_nvidia_quant_ops_probe_one_op(op: str) -> None:
    """在 NVIDIA GPU 上探测英伟达侧量化相关算子名（当前 runner 仍为 FP16 matmul 占位）。"""
    vendor = _gpu_vendor()
    if vendor not in ("nvidia", "hygon"):
        pytest.skip(f"unsupported LUBAN_TEST_GPU_VENDOR={vendor}")
    result = probe_one_op(
        op,
        vendor=vendor,
        timeout_sec=60.0,
        latency_threshold_ms=None,
        mse_threshold=0.05,
    )
    assert result["ok"] is True, result["message"]
    assert result["latency_ms_p50"] is not None
    assert "cuda" in result["message"]


@requires_cuda
def test_gpu_int8_weight_dequant_gemm() -> None:
    """简易 W8A16 路径：INT8 权重反量化后与 FP16 激活做 GEMM（GPU）。"""
    import torch

    device = "cuda:0"
    n, k = 128, 128
    w_i8 = torch.randint(-128, 127, (n, k), device=device, dtype=torch.int8)
    scale = torch.randn(n, device=device, dtype=torch.float16).abs() + 1e-3
    x = torch.randn(k, n, device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    w = w_i8.to(torch.float16) * scale[:, None]
    y = w @ x
    torch.cuda.synchronize()

    assert y.shape == (n, n)
    assert torch.isfinite(y).all()

    # 与 FP32 参考比 MSE，作为「量化算子」数值烟测
    ref = (w_i8.float() * scale[:, None].float()) @ x.float()
    mse = float(torch.mean((y.float() - ref) ** 2).item())
    assert mse < 1.0


@requires_cuda
def test_gpu_hae_probe_profile_nvidia_h20_ops() -> None:
    """端到端：按 nvidia_h20 profile 的 required_ops 在 GPU 上 probe。"""
    from luban_sculpt.hae.engine import HardwareAwareEngine, load_profile_template

    profile = load_profile_template("nvidia_h20")
    ops = list(profile.get("required_ops_probe") or [])
    assert ops, "nvidia_h20 should declare required_ops_probe"

    hae = HardwareAwareEngine(
        "nvidia_h20",
        enable_real_probe=True,
        fallback_on_probe_failure=False,
        mse_threshold=0.05,
    )
    # 注入 nvidia identity，避免 detect 落到 cpu
    hae._identity = {
        "device_name": "NVIDIA H20",
        "vendor": "nvidia",
        "stack": {"cuda": "12.0"},
        "source": "test",
    }
    probe = hae.probe(profile)
    assert probe.ok is True, probe.messages
    assert probe.missing_ops == []
    assert any("cuda" in m for m in probe.messages)
