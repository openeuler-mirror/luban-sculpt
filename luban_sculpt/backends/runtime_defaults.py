"""Profile / vendor 级 compress 运行时默认（用户 recipe 无需写 backend 块）。"""

from __future__ import annotations

from typing import Any

# (vendor, backend) → 非敏感默认；recipe / profile.backends.*.defaults 可覆盖
_VENDOR_BACKEND_DEFAULTS: dict[tuple[str, str], dict[str, Any]] = {
    ("nvidia", "llm_compressor"): {
        "trust_remote_code": True,
        "torch_dtype": "float16",
        "device_map": "cuda:0",
        "pipeline": "sequential",
    },
    ("ascend", "msmodelslim"): {
        "device": "npu",
        "trust_remote_code": True,
    },
    ("hygon", "llm_compressor"): {
        "trust_remote_code": True,
    },
    ("hygon", "hygon"): {
        "trust_remote_code": True,
    },
    ("unknown", "llm_compressor"): {
        "trust_remote_code": True,
    },
}

_BACKEND_GLOBAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "msmodelslim": {"trust_remote_code": True, "device": "npu"},
    "gptq": {"trust_remote_code": True},
    "awq": {"trust_remote_code": True},
}


def merge_backend_runtime_options(
    backend: str,
    profile: dict[str, Any],
    user_opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """合并 vendor / profile / recipe 三层 backend_options（后者优先）。"""
    vendor = str(profile.get("vendor") or "unknown").lower()
    merged: dict[str, Any] = {}
    merged.update(_BACKEND_GLOBAL_DEFAULTS.get(backend, {}))
    merged.update(_VENDOR_BACKEND_DEFAULTS.get((vendor, backend), {}))

    backends_cfg = profile.get("backends") or {}
    bentry = backends_cfg.get(backend)
    if isinstance(bentry, dict):
        defaults = bentry.get("defaults")
        if isinstance(defaults, dict):
            merged.update(defaults)

    merged.update(user_opts or {})
    return merged
