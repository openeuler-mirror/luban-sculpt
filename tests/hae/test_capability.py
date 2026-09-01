"""profiles → HardwareCapability 查表测试。"""

from __future__ import annotations

from luban_sculpt.hae.capability import (
    build_hardware_capability,
    clear_capability_index_cache,
    lookup_device_caps,
    match_profile_id,
)


def setup_function() -> None:
    clear_capability_index_cache()


def test_lookup_h20_from_profile() -> None:
    caps = lookup_device_caps("NVIDIA H20", "nvidia")
    assert caps["fp8_native"] is True
    assert caps["compute_capability"] == "9.0"
    assert caps["peak_tflops_fp16"] == 148.0
    assert match_profile_id("NVIDIA H20") == "nvidia_h20"


def test_lookup_910b() -> None:
    assert match_profile_id("Ascend 910B") == "ascend_910b"
    caps = lookup_device_caps("Ascend 910B", "ascend")
    assert caps["fp8_native"] is False
    assert caps["peak_tflops_fp16"] == 360.0


def test_lookup_hygon_dcu() -> None:
    assert match_profile_id("Hygon DCU") == "hygon_dcu"
    caps = lookup_device_caps("Hygon DCU", "hygon")
    assert caps["fp8_native"] is False


def test_build_hardware_capability_uses_table() -> None:
    cap = build_hardware_capability(
        {"device_name": "H20", "vendor": "nvidia", "stack": {}, "source": "env"}
    )
    assert cap.fp8_native is True
    assert cap.compute_capability == "9.0"
    assert cap.memory_bandwidth_gbps == 4000.0
