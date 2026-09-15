"""Profile + 精度 → abstract_scheme / compress backend（HAE 智能路由）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from luban_sculpt.backends.runtime_defaults import merge_backend_runtime_options
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


def _resolve_scheme_cfg(abstract_scheme: str, schemes_cfg: dict) -> tuple[str, dict | None]:
    from luban_sculpt.validate.quant_capability import _resolve_scheme_cfg as _impl

    return _impl(abstract_scheme, schemes_cfg)

AUTO_BACKEND = "auto"

# 与 pipeline.config._STAGE_COMPRESS_OVERLAY_KEYS 对齐
_COMPRESS_OVERLAY_KEYS = frozenset(
    {
        "modifiers",
        "modifier_chain",
        "observer",
        "weight_observer",
        "input_observer",
        "output_observer",
        "targets",
    }
)
_BACKENDS_WITH_LC_OVERLAY = frozenset({"llm_compressor", "hygon"})

# 用户面向的「量化精度」→ 各 vendor 的 profile scheme 键
_PRECISION_BY_VENDOR: dict[str, dict[str, str]] = {
    "nvidia": {
        "fp8_dynamic": "fp8_dynamic",
        "fp8": "fp8_dynamic",
        "w8a8_fp8": "w8a8_fp8",
        "w8a8": "w8a8_fp8",
        "fp8_block": "fp8_block",
        "w4a16": "w4a16",
        "w4a8_fp8": "w4a8_fp8",
        "w4_gptq": "w4_gptq",
        "w8a8_int8": "w8a8_int8",
    },
    "ascend": {
        "fp8_dynamic": "ascend_w8a8",
        "fp8": "ascend_w8a8",
        "w8a8": "ascend_w8a8",
        "fp8_block": "ascend_w4a8",
        "w4a8": "ascend_w4a8",
        "w4a16": "ascend_w4a8",
        "ascend_fp8": "ascend_fp8",
    },
    "hygon": {
        "fp8_dynamic": "fp8_dynamic",
        "w8a8": "hygon_w8a8_gptq",
        "w4a16": "hygon_w4a16_awq",
    },
    "unknown": {
        "fp8_dynamic": "fp8_dynamic",
        "w8a8": "w8a8_fp8",
        "w4a16": "w4a16",
        "w4_gptq": "w4_gptq",
    },
}


@dataclass(frozen=True)
class CompressRoute:
    """HAE / CLI 决策结果：打到哪条 compress 路径。"""

    profile_id: str
    vendor: str
    precision: str
    abstract_scheme: str
    backend: str
    infer_runtime: str | None = None


def _vendor_key(profile: dict[str, Any]) -> str:
    return str(profile.get("vendor") or "unknown").lower()


def normalize_precision(raw: str) -> str:
    key = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "fp8_dynamic": "fp8_dynamic",
        "fp8dynamic": "fp8_dynamic",
        "dynamic_fp8": "fp8_dynamic",
        "block_fp8": "fp8_block",
        "int8": "w8a8_int8",
    }
    return aliases.get(key, key)


def resolve_scheme_for_profile(
    profile: dict[str, Any],
    precision_or_scheme: str,
) -> str:
    """将 recipe 的 precision 或 abstract_scheme 解析为 profile.schemes 中的键。"""
    raw = str(precision_or_scheme).strip()
    if not raw:
        raise ValueError("empty precision / abstract_scheme")

    schemes_cfg = profile.get("schemes", {}) or {}
    if raw in schemes_cfg:
        return raw

    _resolved, cfg = _resolve_scheme_cfg(raw, schemes_cfg)
    if cfg is not None:
        return _resolved

    vendor = _vendor_key(profile)
    precision = normalize_precision(raw)
    mapped = _PRECISION_BY_VENDOR.get(vendor, {}).get(precision)
    if mapped and mapped in schemes_cfg:
        logger.info(
            "precision %s → scheme %s (vendor=%s)",
            precision,
            mapped,
            vendor,
        )
        return mapped

    supported = list(schemes_cfg.keys())
    raise ValueError(
        f"cannot map precision/scheme {raw!r} for profile {profile.get('id')!r}; "
        f"supported schemes: {supported}"
    )


def resolve_compress_backend(
    profile: dict[str, Any],
    abstract_scheme: str,
    backend_hint: str | None = AUTO_BACKEND,
) -> str:
    """按 profile.schemes[scheme].compress.backend 选 compress 后端；hint 非 auto 时优先用户指定。"""
    hint = (backend_hint or AUTO_BACKEND).strip().lower()
    schemes_cfg = profile.get("schemes", {}) or {}
    resolved_scheme, scheme_cfg = _resolve_scheme_cfg(abstract_scheme, schemes_cfg)
    if scheme_cfg is None:
        raise ValueError(
            f"abstract_scheme {abstract_scheme!r} not in profile {profile.get('id')!r}"
        )

    compress = scheme_cfg.get("compress", {}) or {}
    wired = compress.get("backend")

    if hint and hint != AUTO_BACKEND:
        backends = profile.get("backends", {}) or {}
        if hint not in backends or not backends[hint].get("enabled", True):
            raise ValueError(
                f"backend {hint!r} not enabled on profile {profile.get('id')!r}"
            )
        if wired and wired != hint:
            logger.warning(
                "backend override %s (profile wires %s for scheme %s)",
                hint,
                wired,
                resolved_scheme,
            )
        return hint

    if wired:
        return str(wired)

    backends = profile.get("backends", {}) or {}
    for name, cfg in backends.items():
        if isinstance(cfg, dict) and cfg.get("enabled", False):
            logger.info(
                "scheme %s has no compress.backend; fallback first enabled backend=%s",
                resolved_scheme,
                name,
            )
            return str(name)

    return "llm_compressor"


def merge_compress_scheme_defaults(
    backend: str,
    abstract_scheme: str,
    profile: dict[str, Any],
    backend_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """把 profile scheme 上的 default_quant_type / default_scheme 写入 backend_options。"""
    opts = dict(backend_options or {})
    schemes_cfg = profile.get("schemes", {}) or {}
    _name, scheme_cfg = _resolve_scheme_cfg(abstract_scheme, schemes_cfg)
    if not scheme_cfg:
        return opts
    compress = scheme_cfg.get("compress", {}) or {}
    if backend == "msmodelslim":
        opts.setdefault("quant_type", compress.get("default_quant_type"))
    if backend == "llm_compressor":
        opts.setdefault("scheme", compress.get("default_scheme"))
    return {k: v for k, v in opts.items() if v is not None}


def suggest_compress_route(
    profile: dict[str, Any],
    *,
    precision: str = "fp8_dynamic",
    backend_hint: str = AUTO_BACKEND,
) -> CompressRoute:
    """根据已加载的 profile（通常来自 HAE.run）给出 compress 路由建议。"""
    prec = normalize_precision(precision)
    scheme = resolve_scheme_for_profile(profile, prec)
    backend = resolve_compress_backend(profile, scheme, backend_hint)
    schemes_cfg = profile.get("schemes", {}) or {}
    _name, scheme_cfg = _resolve_scheme_cfg(scheme, schemes_cfg)
    infer_runtime = None
    if scheme_cfg:
        infer = scheme_cfg.get("infer") or {}
        infer_runtime = infer.get("infer_runtime") or infer.get("runtime")
    return CompressRoute(
        profile_id=str(profile.get("id", "unknown")),
        vendor=_vendor_key(profile),
        precision=prec,
        abstract_scheme=scheme,
        backend=backend,
        infer_runtime=str(infer_runtime) if infer_runtime else None,
    )


def resolve_quant_block(
    quant: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Recipe ``quant`` 段：解析 scheme + backend，并合并 scheme 默认 backend_options。"""
    q = dict(quant or {})
    precision = q.get("precision")
    abstract = q.get("abstract_scheme") or q.get("scheme")
    if abstract:
        scheme = resolve_scheme_for_profile(profile, str(abstract))
    elif precision:
        scheme = resolve_scheme_for_profile(profile, str(precision))
    else:
        scheme = resolve_scheme_for_profile(profile, "fp8_dynamic")

    backend_hint = q.get("backend", AUTO_BACKEND)
    backend = resolve_compress_backend(profile, scheme, str(backend_hint))

    overlay = q.get("compress_overlay") or {}
    if not isinstance(overlay, dict):
        overlay = {}
    raw_opts = q.get(backend, q.get("backend_options", {}))
    if not isinstance(raw_opts, dict):
        raw_opts = {}
    # overlay 先写，backend 专属块覆盖（兼容旧 recipe 写在 llm_compressor: 下）
    user_opts = {**overlay, **raw_opts}
    if backend not in _BACKENDS_WITH_LC_OVERLAY:
        for key in overlay:
            user_opts.pop(key, None)

    opts = merge_backend_runtime_options(
        backend,
        profile,
        merge_compress_scheme_defaults(backend, scheme, profile, user_opts),
    )
    return scheme, backend, opts
