"""BackendPlan → gptqmodel GPTQConfig."""

from __future__ import annotations

from typing import Any

from luban_sculpt.contracts import BackendPlan

_DEFAULT_BITS = 4
_DEFAULT_GROUP = 128


def build_gptq_config(plan: BackendPlan) -> Any:
    opts = plan.intent.backend_options or {}
    gptq = opts.get("gptq") or opts
    bits = int(gptq.get("bits", _DEFAULT_BITS))
    group_size = int(gptq.get("group_size", _DEFAULT_GROUP))
    sym = gptq.get("sym", True)

    try:
        from gptqmodel import GPTQConfig

        kwargs: dict[str, Any] = {}
        if gptq.get("desc_act") is not None:
            kwargs["desc_act"] = gptq["desc_act"]
        if gptq.get("device"):
            kwargs["device"] = gptq["device"]
        return GPTQConfig(bits=bits, group_size=group_size, sym=sym, **kwargs)
    except ImportError:
        return {
            "GPTQConfig": {
                "bits": bits,
                "group_size": group_size,
                "sym": sym,
            }
        }


def resolve_model_id(plan: BackendPlan) -> str:
    opts = plan.intent.backend_options or {}
    return str(opts.get("model_path") or plan.intent.model_id)
