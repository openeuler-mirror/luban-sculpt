"""Hardware detection: backend API → system CLI → environment variables."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any


def _run_cmd(args: list[str], timeout: float = 5.0) -> str | None:
    if not shutil.which(args[0]):
        return None
    try:
        out = subprocess.check_output(
            args, stderr=subprocess.DEVNULL, timeout=timeout, text=True
        )
        return out.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _detect_via_torch_cuda() -> dict[str, Any] | None:
    try:
        import contextlib
        import io
        import warnings

        buf = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(buf):
                import torch
    except ImportError:
        return None
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return None
    idx = 0
    props = torch.cuda.get_device_properties(idx)
    major, minor = props.major, props.minor
    name = props.name
    stack = {
        "cuda": getattr(torch.version, "cuda", None) or os.environ.get("CUDA_VERSION", "unknown"),
        "driver": os.environ.get("NVIDIA_DRIVER_VERSION", "unknown"),
    }
    return {
        "device_name": name,
        "vendor": "nvidia",
        "stack": {k: str(v) for k, v in stack.items() if v},
        "compute_capability": f"{major}.{minor}",
        "memory_bytes": int(props.total_memory),
        "sm_count": int(getattr(props, "multi_processor_count", 0) or 0) or None,
        "source": "api",
    }


def _detect_via_torch_npu() -> dict[str, Any] | None:
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return None
    name = "ascend"
    try:
        name = torch.npu.get_device_name(0)
    except Exception:
        pass
    stack = {"cann": os.environ.get("CANN_VERSION", "unknown")}
    mem = None
    try:
        free, total = torch.npu.mem_get_info(0)
        mem = int(total)
    except Exception:
        pass
    return {
        "device_name": name,
        "vendor": "ascend",
        "stack": stack,
        "memory_bytes": mem,
        "source": "api",
    }


def _detect_via_torch_hip() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    hip = getattr(torch.version, "hip", None)
    if not hip:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "device_name": props.name,
        "vendor": "hygon",
        "stack": {"dtk": os.environ.get("DTK_VERSION", str(hip))},
        "memory_bytes": int(props.total_memory),
        "sm_count": int(getattr(props, "multi_processor_count", 0) or 0) or None,
        "source": "api",
    }


def _detect_via_smi() -> dict[str, Any] | None:
    npu = _run_cmd(["npu-smi", "info"])
    if npu:
        # crude parse: look for chip name lines
        chip = "ascend"
        m = re.search(r"Chip Name\s*:\s*(\S+)", npu, re.I)
        if m:
            chip = m.group(1)
        return {
            "device_name": chip,
            "vendor": "ascend",
            "stack": {"cann": os.environ.get("CANN_VERSION", "unknown")},
            "source": "smi",
        }

    nv = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if nv:
        line = nv.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if parts else "nvidia"
        driver = parts[1] if len(parts) > 1 else None
        mem_mib = None
        if len(parts) > 2:
            try:
                mem_mib = int(float(parts[2])) * 1024 * 1024
            except ValueError:
                pass
        return {
            "device_name": name,
            "vendor": "nvidia",
            "stack": {
                "cuda": os.environ.get("CUDA_VERSION", "unknown"),
                "driver": driver or os.environ.get("NVIDIA_DRIVER_VERSION", "unknown"),
            },
            "driver_version": driver,
            "memory_bytes": mem_mib,
            "source": "smi",
        }

    rocm = _run_cmd(["rocm-smi", "--showproductname"])
    if rocm:
        return {
            "device_name": rocm.splitlines()[-1].strip() or "dcu",
            "vendor": "hygon",
            "stack": {"dtk": os.environ.get("DTK_VERSION", "unknown")},
            "source": "smi",
        }
    return None


def _detect_via_env() -> dict[str, Any]:
    device_name = os.environ.get("LUBAN_DEVICE_NAME", "unknown")
    vendor = "unknown"
    stack: dict[str, str] = {}

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None or "ascend" in device_name.lower():
        vendor = "ascend"
        stack["cann"] = os.environ.get("CANN_VERSION", "unknown")
    elif os.environ.get("HIP_VISIBLE_DEVICES") is not None or "dcu" in device_name.lower():
        vendor = "hygon"
        stack["dtk"] = os.environ.get("DTK_VERSION", "unknown")
    elif os.environ.get("CUDA_VISIBLE_DEVICES") is not None or "nvidia" in device_name.lower():
        vendor = "nvidia"
        stack["cuda"] = os.environ.get("CUDA_VERSION", "unknown")
        stack["driver"] = os.environ.get("NVIDIA_DRIVER_VERSION", "unknown")
    elif os.environ.get("LUBAN_VENDOR"):
        vendor = os.environ["LUBAN_VENDOR"]

    return {
        "device_name": device_name,
        "vendor": vendor,
        "stack": stack,
        "source": "env",
    }


def detect_hardware() -> dict[str, Any]:
    """优先 API → smi → 环境变量。返回 identity extras（含 source）。"""
    for fn in (_detect_via_torch_cuda, _detect_via_torch_npu, _detect_via_torch_hip):
        hit = fn()
        if hit:
            return hit
    hit = _detect_via_smi()
    if hit:
        return hit
    return _detect_via_env()


def detect_topology_nvml() -> str | None:
    """通过 NVML 推断 NVLink/PCIe；成功返回 ParallelTopology.value。"""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count < 2:
            pynvml.nvmlShutdown()
            return "pcie_switch"
        # any NVLink between 0 and others → treat as mesh-like high-speed fabric
        has_nvlink = False
        handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
        for i in range(1, count):
            try:
                hi = pynvml.nvmlDeviceGetHandleByIndex(i)
                # NVML_FI_DEV_NVLINK_COUNT_* varies by version; probe link state if available
                for link in range(6):
                    try:
                        st = pynvml.nvmlDeviceGetNvLinkState(handle0, link)
                        if st:
                            has_nvlink = True
                            break
                    except pynvml.NVMLError:
                        continue
                if has_nvlink:
                    break
                _ = hi
            except pynvml.NVMLError:
                continue
        pynvml.nvmlShutdown()
        return "pcie_switch" if not has_nvlink else "hccs_mesh"  # NVLink ~ mesh-class
    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return None


def detect_topology_cann() -> str | None:
    """CANN/Ascend 拓扑探测占位：有多卡环境时默认 HCCS ring。"""
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is None and not shutil.which("npu-smi"):
        return None
    info = _run_cmd(["npu-smi", "info", "-t", "topo"])
    if info is None:
        # device present but no topo subcommand
        if shutil.which("npu-smi"):
            return "hccs_ring"
        return None
    lower = info.lower()
    if "mesh" in lower:
        return "hccs_mesh"
    if "ring" in lower or "hccs" in lower:
        return "hccs_ring"
    if "pcie" in lower:
        return "pcie_switch"
    return "hccs_ring"
