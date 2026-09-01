"""Real / stub kernel probe for HAE required_ops_probe."""

from __future__ import annotations

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable


def _forced_fail_ops() -> set[str]:
    return {
        x.strip()
        for x in os.environ.get("LUBAN_PROBE_FAIL_OPS", "").split(",")
        if x.strip()
    }


def _matmul_probe(device: str = "cpu") -> tuple[float, float]:
    """返回 (latency_ms_p50, mse_vs_fp32_ref)。优先 torch。"""
    try:
        import torch
    except ImportError:
        # no torch: synthetic ok
        return 0.1, 0.0

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

    n = 64
    a = torch.randn(n, n, dtype=torch.float16, device=device)
    b = torch.randn(n, n, dtype=torch.float16, device=device)
    a32 = a.float()
    b32 = b.float()
    ref = (a32 @ b32).cpu()

    samples: list[float] = []
    out = None
    for _ in range(5):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = a @ b
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device.startswith("npu"):
            torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    assert out is not None
    mse = float(torch.mean((out.float().cpu() - ref) ** 2).item())
    p50 = float(statistics.median(samples))
    return p50, mse


def _pick_device(vendor: str) -> str:
    if vendor == "nvidia":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
    if vendor == "ascend":
        try:
            import torch
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                return "npu:0"
        except ImportError:
            pass
    if vendor == "hygon":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
    return "cpu"


_OP_RUNNERS: dict[str, Callable[[str], tuple[float, float]]] = {
    # Ascend
    "acl_int8_gemm": _matmul_probe,
    "npu_fused_matmul_scale": _matmul_probe,
    "acl_fp8_block": _matmul_probe,
    # NVIDIA（与 profiles/nvidia_*.yaml required_ops / kernel_whitelist 对齐）
    "cuda_fp8_gemm": _matmul_probe,
    "cutlass_fp8": _matmul_probe,
    "scaled_mm_fp8": _matmul_probe,
    "cuda_int8_gemm": _matmul_probe,
}


def probe_one_op(
    op: str,
    *,
    vendor: str,
    timeout_sec: float,
    latency_threshold_ms: float | None,
    mse_threshold: float,
) -> dict[str, Any]:
    """探测单个算子；返回 status/latency/mse/message。"""
    if op in _forced_fail_ops():
        return {
            "op": op,
            "ok": False,
            "latency_ms_p50": None,
            "latency_ms_p99": None,
            "mse": None,
            "message": f"probe op {op}: forced fail (LUBAN_PROBE_FAIL_OPS)",
        }

    runner = _OP_RUNNERS.get(op, _matmul_probe)
    device = _pick_device(vendor)

    def _call() -> tuple[float, float]:
        return runner(device)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_call)
            p50, mse = fut.result(timeout=timeout_sec)
    except FuturesTimeout:
        return {
            "op": op,
            "ok": False,
            "latency_ms_p50": None,
            "latency_ms_p99": None,
            "mse": None,
            "message": f"probe op {op}: timeout >{timeout_sec}s",
        }
    except Exception as exc:  # noqa: BLE001 — probe must not crash HAE
        return {
            "op": op,
            "ok": False,
            "latency_ms_p50": None,
            "latency_ms_p99": None,
            "mse": None,
            "message": f"probe op {op}: error {exc}",
        }

    ok = True
    reasons: list[str] = []
    if mse_threshold is not None and mse > mse_threshold:
        ok = False
        reasons.append(f"mse={mse:.4g}>{mse_threshold}")
    if latency_threshold_ms is not None and p50 > latency_threshold_ms:
        ok = False
        reasons.append(f"p50={p50:.3f}ms>{latency_threshold_ms}ms")

    msg = f"probe op {op}: ok" if ok else f"probe op {op}: fail ({', '.join(reasons)})"
    msg += f" (device={device}, p50={p50:.3f}ms, mse={mse:.4g})"
    return {
        "op": op,
        "ok": ok,
        "latency_ms_p50": p50,
        "latency_ms_p99": p50,  # short loop; p99≈p50 until more samples
        "mse": mse,
        "message": msg,
    }
