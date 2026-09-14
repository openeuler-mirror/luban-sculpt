"""Custom backend: optional user command in plan backend_options."""

from __future__ import annotations

from luban_sculpt.contracts import BackendPlan


def custom_compressor_configured(plan: BackendPlan) -> bool:
    opts = plan.intent.backend_options or {}
    custom = opts.get("custom")
    return isinstance(custom, dict) and bool(custom.get("command") or custom.get("script"))
