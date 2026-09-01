"""Compress-前：量化意图 vs profile / probe / 芯片能力。"""

from __future__ import annotations

from typing import Any

from luban_sculpt.backends.msmodelslim.chips import scheme_implies_fp8
from luban_sculpt.contracts import HwDecision, ProbeResult, QuantIntent
from luban_sculpt.hae.profile_fields import (
    infer_expected_deploy,
    profile_fp8_native,
    profile_soc_key,
)
from luban_sculpt.log import get_logger

logger = get_logger(__name__)

# 旧 profile 仅有 w4a16_g32 时，把 w4a16 视作别名
_SCHEME_ALIASES: dict[str, tuple[str, ...]] = {
    "w4a16": ("w4a16_g32", "w4a16"),
    "w4a16_g32": ("w4a16_g32", "w4a16"),
}


class QuantCapabilityError(RuntimeError):
    """Profile / 探测 / 芯片能力与量化意图不匹配时抛出。"""


def _resolve_scheme_cfg(
    abstract_scheme: str, schemes_cfg: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """返回 (实际命中的 scheme 名, cfg)；支持别名回退。"""
    if abstract_scheme in schemes_cfg:
        return abstract_scheme, schemes_cfg[abstract_scheme]
    for alt in _SCHEME_ALIASES.get(abstract_scheme, ()):
        if alt in schemes_cfg:
            logger.info(
                "scheme alias %s → profile key %s",
                abstract_scheme,
                alt,
            )
            return alt, schemes_cfg[alt]
    return abstract_scheme, None


def validate_intent_profile(
    intent: QuantIntent, hw: HwDecision, profile: dict[str, Any], probe: ProbeResult
) -> None:
    """校验 Intent 与 profile/probe 是否匹配：probe、scheme、backend、deploy。"""
    if not probe.ok:
        msg = f"Probe failed: {probe.missing_ops}. Set LUBAN_* env or fix stack."
        logger.error(msg)
        raise QuantCapabilityError(msg)

    schemes_cfg = profile.get("schemes", {})
    resolved_name, scheme_cfg = _resolve_scheme_cfg(intent.abstract_scheme, schemes_cfg)
    if scheme_cfg is None:
        supported = list(schemes_cfg.keys())
        msg = (
            f"abstract_scheme {intent.abstract_scheme!r} not in profile; "
            f"supported: {supported}"
        )
        logger.error(msg)
        raise QuantCapabilityError(msg)
    # 别名命中时写回，后续 compile / manifest 用 profile 真实 key
    if resolved_name != intent.abstract_scheme:
        intent.abstract_scheme = resolved_name

    compress = scheme_cfg.get("compress", {})
    allowed_backend = compress.get("backend")
    if allowed_backend and intent.backend != allowed_backend:
        backends = profile.get("backends", {})
        if intent.backend not in backends or not backends[intent.backend].get(
            "enabled", False
        ):
            msg = (
                f"backend {intent.backend!r} not enabled for scheme "
                f"{intent.abstract_scheme!r}"
            )
            logger.error(msg)
            raise QuantCapabilityError(msg)

    stack_gates = profile.get("stack_gates", {})
    if intent.abstract_scheme in stack_gates and probe.missing_ops:
        msg = (
            f"Scheme {intent.abstract_scheme} blocked by stack gate: {probe.missing_ops}"
        )
        logger.error(msg)
        raise QuantCapabilityError(msg)

    deploy = intent.deploy_target
    expected = infer_expected_deploy(scheme_cfg)
    if expected and deploy != expected:
        msg = f"deploy_target {deploy!r} != profile infer deploy {expected!r}"
        logger.error(msg)
        raise QuantCapabilityError(msg)

    logger.info(
        "validate_intent_profile passed scheme=%s backend=%s deploy=%s profile=%s",
        intent.abstract_scheme,
        intent.backend,
        intent.deploy_target,
        hw.profile_id,
    )


def validate_quant_capability(
    intent: QuantIntent, hw: HwDecision, profile: dict[str, Any], probe: ProbeResult
) -> None:
    """compress 前总检查：``validate_intent_profile`` + FP8 / msmodelslim 约束。"""
    validate_intent_profile(intent, hw, profile, probe)
    schemes = profile.get("schemes", {})
    if intent.abstract_scheme not in schemes:
        msg = f"Unknown scheme: {intent.abstract_scheme}"
        logger.error(msg)
        raise ValueError(msg)

    vendor = (profile.get("vendor") or hw.vendor or "unknown").lower()
    enforce_fp8 = vendor in ("ascend", "nvidia", "hygon")
    fp8_native = profile_fp8_native(profile)
    soc_label = profile_soc_key(profile) or hw.profile_id
    if enforce_fp8 and not fp8_native and scheme_implies_fp8(intent.abstract_scheme):
        msg = (
            f"Scheme {intent.abstract_scheme!r} requires FP8 native compute; "
            f"profile {hw.profile_id} ({soc_label}) is INT8/W8A8 only."
        )
        logger.error(msg)
        raise QuantCapabilityError(msg)
    if intent.backend == "msmodelslim":
        qt = (intent.backend_options or {}).get("quant_type", "")
        if qt and str(qt).lower() == "fp8" and not fp8_native:
            msg = f"quant_type fp8 blocked on {hw.profile_id}"
            logger.error(msg)
            raise QuantCapabilityError(msg)
    logger.info(
        "validate_quant_capability passed scheme=%s fp8_native=%s vendor=%s",
        intent.abstract_scheme,
        fp8_native,
        vendor,
    )
