"""Domestic accelerator platform stub — register when wiring into vLLM.

Reads ``manifest.json`` ``profile_id`` / ``producer`` to select kernels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class DomesticPlatform:
    """Minimal stand-in for vLLM Platform; does not import vllm by default."""

    device_name: str = "domestic"
    dispatch_key: str = "PrivateUse1"

    @classmethod
    def from_manifest(cls, model_path: str | Path) -> "DomesticPlatform":
        inst = cls()
        path = Path(model_path) / "manifest.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            inst.device_name = data.get("profile_id", "domestic")
            inst._manifest = data  # noqa: SLF001
        else:
            inst._manifest = {}
        return inst

    @classmethod
    def detect(cls) -> bool:
        if os.environ.get("LUBAN_FORCE_DOMESTIC"):
            return True
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None:
            return True
        if os.environ.get("HIP_VISIBLE_DEVICES") is not None and os.environ.get(
            "LUBAN_VENDOR"
        ) == "hygon":
            return True
        return False

    def kernel_whitelist(self) -> list[str]:
        manifest: dict[str, Any] = getattr(self, "_manifest", {})
        hw = manifest.get("hw_decision") or {}
        return list(hw.get("probe_passed_ops") or [])
