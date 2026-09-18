"""BackendPlan → llm-compressor modifier 链（对齐 llama3_example 的 algorithm 选择）。"""

from __future__ import annotations

import inspect
from typing import Any

from luban_sculpt.backends.compress_spec import resolve_compress_spec
from luban_sculpt.backends.llm_compressor.check import (
    is_llm_compressor_available,
    get_llm_compressor_quantization_modifier_class,
)
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
    """从 ``BackendPlan`` 组装 base modifier 链（供 ``ModifierManager.base_builder``）。

    解析 scheme/targets/ignore 后二选一：

    - **dry-run**（无 LC 或仅编排验证）→ ``_build_base_modifiers_dry_run``
    - **真实 oneshot** → ``_build_base_modifiers_instances``
    """
    opts = plan.intent.backend_options or {}
    lc = opts.get("llm_compressor") or opts
    spec = resolve_compress_spec(plan.intent.abstract_scheme, compressor_options=lc)
    targets = lc.get("targets", "Linear")
    ignore = plan.intent.ignore or lc.get("ignore") or ["lm_head"]

    if not is_llm_compressor_available():
        return _build_base_modifiers_dry_run(spec, targets, ignore, lc)

    quant_mod_cls = get_llm_compressor_quantization_modifier_class()
    if quant_mod_cls is None:
        return _build_base_modifiers_dry_run(spec, targets, ignore, lc)

    return _build_base_modifiers_instances(
        quant_mod_cls, spec, targets, ignore, lc
    )


def _build_base_modifiers_dry_run(
    spec: Any, targets: Any, ignore: list[str], lc: dict[str, Any]
) -> list[Any]:
    """``build_base_modifiers`` dry-run：可序列化 dict 链（与 instances 同 algorithm 分支）。"""
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


def _build_base_modifiers_instances(
    quantization_modifier: type[Any],
    spec: Any,
    targets: Any,
    ignore: list[str],
    lc: dict[str, Any],
) -> list[Any]:
    """``build_base_modifiers`` 真实 oneshot：llmcompressor Modifier 实例链。"""
    observer_overrides = _extract_observer_overrides(lc)
    if spec.algorithm == "gptq":
        gptq_modifier = _import_gptq()
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
        filtered = _filter_kwargs(gptq_modifier, kwargs)
        if "offload_hessians" in getattr(gptq_modifier, "model_fields", {}) and (
            "offload_hessians" not in filtered
        ):
            filtered["offload_hessians"] = True
        logger.info("GPTQModifier kwargs=%s", sorted(filtered.keys()))
        return [gptq_modifier(**filtered)]
    if spec.algorithm == "awq":
        awq_modifier = _import_awq()
        quant_kwargs: dict[str, Any] = {
            "targets": targets,
            "scheme": spec.scheme,
            "ignore": ignore,
        }
        quant_kwargs.update(observer_overrides)
        filtered_quant = _filter_kwargs(quantization_modifier, quant_kwargs)
        return [
            awq_modifier(duo_scaling=True),
            quantization_modifier(**filtered_quant),
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
    filtered = _filter_kwargs(quantization_modifier, kwargs)

    logger.info(
        "QuantizationModifier scheme=%s kwargs=%s observer_overrides=%s",
        spec.scheme,
        sorted(filtered.keys()),
        observer_overrides,
    )

    return [quantization_modifier(**filtered)]


def _import_gptq() -> Any:
    from llmcompressor.modifiers.gptq import GPTQModifier

    return GPTQModifier


def _import_awq() -> Any:
    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier
    except ImportError:
        from llmcompressor.modifiers.awq import AWQModifier  # type: ignore[attr-defined]
    return AWQModifier
