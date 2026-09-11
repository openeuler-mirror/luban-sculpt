"""BackendPlan → llm-compressor modifier 链（对齐 llama3_example 的 algorithm 选择）。"""

from __future__ import annotations

import inspect
from typing import Any

from luban_sculpt.backends.llm_compressor.probe import probe_llmcompressor
from luban_sculpt.backends.llm_compressor.scheme_map import resolve_compress_spec
from luban_sculpt.contracts import BackendPlan
from luban_sculpt.log import get_logger

logger = get_logger(__name__)

# GPTQModifier 可选内存相关字段（按版本探测）
_GPTQ_OPTIONAL_KEYS = (
    "dampening_frac",
    "actorder",
    "offload_hessians",
    "sequential_update",
)
_OBSERVER_OPTIONAL_KEYS = (
    "weight_observer",
    "input_observer",
    "output_observer",
)


def _extract_observer_overrides(lc: dict[str, Any]) -> dict[str, Any]:
    """提取 llm-compressor QuantizationModifier 的 Observer 参数。

    支持两种互斥写法：

    1. observer:
         weights: xxx
         input: xxx

    2. weight_observer / input_observer / output_observer
    """
    observer = lc.get("observer")

    individual = {
        key: lc[key]
        for key in _OBSERVER_OPTIONAL_KEYS
        if lc.get(key) is not None
    }

    if observer is not None and individual:
        raise ValueError(
            "不能同时配置 `observer` 字典和 "
            "`weight_observer/input_observer/output_observer`"
        )

    if observer is not None:
        if not isinstance(observer, dict):
            raise TypeError(
                "`llm_compressor.observer` 必须是字典，例如："
                "{'weights': 'luban_ema_absmax'}"
            )
        return {"observer": dict(observer)}

    return individual



def build_base_modifiers(plan: BackendPlan) -> list[Any]:
    """按 abstract_scheme / CompressSpec 生成 GPTQ、AWQ 或 QuantizationModifier。"""
    opts = plan.intent.backend_options or {}
    lc = opts.get("llm_compressor") or opts
    spec = resolve_compress_spec(plan.intent.abstract_scheme, lc_override=lc)
    targets = lc.get("targets", "Linear")
    ignore = plan.intent.ignore or lc.get("ignore") or ["lm_head"]

    state = probe_llmcompressor()
    if not state.get("available"):
        return _stub_modifiers(spec, targets, ignore, lc)

    return _live_modifiers(state, spec, targets, ignore, lc)


def _stub_modifiers(
    spec: Any, targets: Any, ignore: list[str], lc: dict[str, Any]
) -> list[Any]:
    """无 llmcompressor 时的 JSON 可序列化 stub（dry-run）。"""
    observer_overrides = _extract_observer_overrides(lc)
    if spec.algorithm == "gptq":
        body: dict[str, Any] = {
            "targets": targets,
            "scheme": spec.scheme,
            "ignore": ignore,
        }
        body.update(observer_overrides)
        if spec.block_size is not None:
            body["block_size"] = spec.block_size
        for key in _GPTQ_OPTIONAL_KEYS:
            if key in lc:
                body[key] = lc[key]
        return [{"GPTQModifier": body}]
    if spec.algorithm == "awq":
        quant_body: dict[str, Any] = {
            "targets": targets,
            "scheme": spec.scheme,
            "ignore": ignore,
        }
        quant_body.update(observer_overrides)

        return [
            {
                "AWQModifier": {
                    "duo_scaling": True,
                }
            },
            {
                "QuantizationModifier": quant_body,
            },
        ]

    body: dict[str, Any] = {
        "targets": targets,
        "scheme": spec.scheme,
        "ignore": ignore,
    }
    body.update(observer_overrides)
    # QuantizationModifier 不接受 block_size（仅 GPTQModifier 使用）
    return [{"QuantizationModifier": body}]


def _filter_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """只保留构造函数 / pydantic 模型接受的字段。"""
    try:
        fields = getattr(cls, "model_fields", None)
        if fields is not None:
            return {k: v for k, v in kwargs.items() if k in fields}
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters) - {"self", "args", "kwargs"}
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:  # noqa: BLE001
        return kwargs


def _live_modifiers(
    state: dict[str, Any],
    spec: Any,
    targets: Any,
    ignore: list[str],
    lc: dict[str, Any],
) -> list[Any]:
    QuantizationModifier = state["QuantizationModifier"]
    observer_overrides = _extract_observer_overrides(lc)
    if spec.algorithm == "gptq":
        GPTQModifier = _import_gptq()
        kwargs: dict[str, Any] = {
            "targets": targets,
            "scheme": spec.scheme,
            "ignore": ignore,
        }
        kwargs.update(observer_overrides)
        if spec.block_size is not None:
            kwargs["block_size"] = spec.block_size
        # 默认 dampening，减轻数值/内存尖峰
        kwargs.setdefault("dampening_frac", lc.get("dampening_frac", 0.01))
        for key in _GPTQ_OPTIONAL_KEYS:
            if key in lc:
                kwargs[key] = lc[key]
        # 新版本若支持 offload_hessians，默认打开（省主机内存）
        filtered = _filter_kwargs(GPTQModifier, kwargs)
        if "offload_hessians" in getattr(GPTQModifier, "model_fields", {}) and (
            "offload_hessians" not in filtered
        ):
            filtered["offload_hessians"] = True
        logger.info("GPTQModifier kwargs=%s", sorted(filtered.keys()))
        return [GPTQModifier(**filtered)]
    if spec.algorithm == "awq":
        AWQModifier = _import_awq()
        quant_kwargs: dict[str, Any] = {
            "targets": targets,
            "scheme": spec.scheme,
            "ignore": ignore,
        }
        quant_kwargs.update(observer_overrides)
        filtered_quant = _filter_kwargs(QuantizationModifier, quant_kwargs)
        return [
            AWQModifier(duo_scaling=True),
            QuantizationModifier(**filtered_quant),
        ]
    kwargs = {
        "targets": targets,
        "scheme": spec.scheme,
        "ignore": ignore,
    }

    # 这里是真正把 YAML 中的 Observer 参数传给 QuantizationModifier。
    kwargs.update(observer_overrides)
    # FP8_BLOCK 等 scheme 自带 block 语义；勿传 block_size（pydantic extra_forbidden）
    if spec.block_size is not None:
        logger.info(
            "ignore block_size=%s for QuantizationModifier (scheme=%s)",
            spec.block_size,
            spec.scheme,
        )
    filtered = _filter_kwargs(QuantizationModifier, kwargs)

    logger.info(
        "QuantizationModifier scheme=%s kwargs=%s observer_overrides=%s",
        spec.scheme,
        sorted(filtered.keys()),
        observer_overrides,
    )

    return [QuantizationModifier(**filtered)]


def _import_gptq() -> Any:
    from llmcompressor.modifiers.gptq import GPTQModifier

    return GPTQModifier


def _import_awq() -> Any:
    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier
    except ImportError:
        from llmcompressor.modifiers.awq import AWQModifier  # type: ignore[attr-defined]
    return AWQModifier
