"""Device model → capability mapping for HAE（数据源：profiles/*.yaml）。"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from luban_sculpt.contracts import HardwareCapability
from luban_sculpt.hae.profile_fields import profile_match_keys


def _profiles_dir() -> Path:
    from importlib.resources import files

    return Path(str(files("luban_sculpt").joinpath("profiles")))


def _caps_from_profile(doc: dict[str, Any]) -> dict[str, Any]:
    """从 profile 的 capability / chip / gpu 块抽取能力字段。"""
    cap = dict(doc.get("capability") or {})
    chip = doc.get("chip") or {}
    gpu = doc.get("gpu") or {}

    if "fp8_native" not in cap:
        if "fp8_native" in chip:
            cap["fp8_native"] = chip["fp8_native"]
        elif "fp8_native" in gpu:
            cap["fp8_native"] = gpu["fp8_native"]
    if "compute_capability" not in cap and gpu.get("compute_capability") is not None:
        cap["compute_capability"] = gpu["compute_capability"]
    if "peak_tflops_fp16" not in cap and chip.get("fp16_tflops") is not None:
        # 保留数值化 capability；字符串描述仍在 chip.fp16_tflops
        pass
    return cap


def _match_keys_for(doc: dict[str, Any]) -> list[str]:
    return profile_match_keys(doc)


@functools.lru_cache(maxsize=1)
def _capability_index() -> tuple[tuple[str, tuple[tuple[str, Any], ...], str], ...]:
    """(match_key, caps_items, profile_id) 按 match_key 长度降序，避免 910 抢先于 910c。"""
    rows: list[tuple[str, dict[str, Any], str]] = []
    root = _profiles_dir()
    if not root.is_dir():
        return tuple()
    for path in sorted(root.glob("*.yaml")):
        with path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            continue
        pid = str(doc.get("id") or path.stem)
        caps = _caps_from_profile(doc)
        for key in _match_keys_for(doc):
            rows.append((key, caps, pid))
    rows.sort(key=lambda r: len(r[0]), reverse=True)
    return tuple((k, tuple(sorted(c.items())), p) for k, c, p in rows)


def clear_capability_index_cache() -> None:
    """测试或热加载时可清缓存。"""
    _capability_index.cache_clear()


def lookup_device_caps(device_name: str, vendor: str) -> dict[str, Any]:
    """按设备名 match_keys 查 profiles，返回 fp8 / 带宽 / 算力等字段。"""
    lower = (device_name or "").lower()
    for key, caps_items, _pid in _capability_index():
        if key and key in lower:
            return dict(caps_items)
    if vendor == "nvidia":
        return {"fp8_native": False, "compute_capability": None}
    if vendor in ("ascend", "hygon"):
        return {"fp8_native": False}
    return {"fp8_native": False}


def match_profile_id(device_name: str) -> str | None:
    """设备名命中 match_keys 时返回对应 profile id。"""
    lower = (device_name or "").lower()
    for key, _caps, pid in _capability_index():
        if key and key in lower:
            return pid
    return None


def build_hardware_capability(
    identity: dict[str, Any],
    *,
    extras: dict[str, Any] | None = None,
) -> HardwareCapability:
    """根据 detect() identity（及可选 extras）构建 HardwareCapability。"""
    extras = extras or {}
    device_name = str(extras.get("device_name") or identity.get("device_name") or "unknown")
    vendor = str(extras.get("vendor") or identity.get("vendor") or "unknown")
    stack = dict(identity.get("stack") or {})
    stack.update(extras.get("stack") or {})

    table = lookup_device_caps(device_name, vendor)
    cc = extras.get("compute_capability") or table.get("compute_capability")
    fp8 = extras.get("fp8_native")
    if fp8 is None:
        fp8 = bool(table.get("fp8_native", False))

    bw = extras.get("memory_bandwidth_gbps", table.get("memory_bandwidth_gbps"))
    tflops = extras.get("peak_tflops_fp16", table.get("peak_tflops_fp16"))

    return HardwareCapability(
        device_name=device_name,
        vendor=vendor,
        driver_version=extras.get("driver_version") or stack.get("driver"),
        compute_capability=str(cc) if cc is not None else None,
        memory_bytes=extras.get("memory_bytes"),
        sm_count=extras.get("sm_count"),
        fp8_native=bool(fp8),
        memory_bandwidth_gbps=float(bw) if bw is not None else None,
        peak_tflops_fp16=float(tflops) if tflops is not None else None,
        stack=stack,
        source=str(extras.get("source") or identity.get("source") or "env"),
    )
