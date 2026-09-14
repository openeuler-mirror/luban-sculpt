"""Hygon backend：llm-compressor 压缩 + enginex-hygon-vllm infer sidecar。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from luban_sculpt.backends.backend_util import run_with_hal
from luban_sculpt.backends.base import QuantBackend
from luban_sculpt.backends.hygon.check import (
    build_hygon_infer_payload,
    lc_options_from_hygon,
    resolve_hygon_spec,
)
from luban_sculpt.backends.oneshot_hooks import run_post_oneshot, run_pre_oneshot
from luban_sculpt.contracts import QuantizedArtifact
from luban_sculpt.contracts import BackendPlan, QuantIntent
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


def plan_for_llm_compressor(plan: BackendPlan) -> tuple[BackendPlan, Any]:
    """注入 LC 压缩参数，返回 (给 llm_compressor 的 plan, HygonCompressSpec)。"""
    opts = dict(plan.intent.backend_options or {})
    hygon_spec = resolve_hygon_spec(plan.intent.abstract_scheme, override=opts)
    lc = lc_options_from_hygon(hygon_spec, opts)
    new_opts = {**opts, "llm_compressor": lc}
    # 顶层也放一份，方便 resolve_compress_spec(compressor_options=opts)
    for k, v in lc.items():
        new_opts.setdefault(k, v)
    intent = QuantIntent(
        model_id=plan.intent.model_id,
        backend="llm_compressor",
        abstract_scheme=plan.intent.abstract_scheme,
        deploy_target=plan.intent.deploy_target,
        arch_snapshot=plan.intent.arch_snapshot,
        ignore=list(plan.intent.ignore),
        calib=dict(plan.intent.calib),
        backend_options=new_opts,
    )
    return (
        BackendPlan(intent=intent, hw=plan.hw, export_format=plan.export_format),
        hygon_spec,
    )


def write_hygon_infer_sidecar(
    save_dir: Path,
    *,
    infer_quantization: str,
    note: str = "",
) -> Path:
    """写出 hygon_infer.json。"""
    payload = build_hygon_infer_payload(
        str(save_dir),
        infer_quantization,
        note=note,
    )
    path = save_dir / "hygon_infer.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("hygon infer sidecar → %s serve=%s", path, payload.get("serve_example"))
    return path


def run_hygon_quant(plan: BackendPlan, output_dir: Path) -> dict[str, Any]:
    """委托 llm_compressor oneshot，再写海光推理 sidecar。"""
    from luban_sculpt.backends.llm_compressor.runner import run_llm_compressor_oneshot

    lc_plan, hygon_spec = plan_for_llm_compressor(plan)
    logger.info(
        "hygon quant scheme=%s infer=%s lc_algo=%s lc_scheme=%s",
        plan.intent.abstract_scheme,
        hygon_spec.infer_quantization,
        hygon_spec.compress.algorithm,
        hygon_spec.compress.scheme,
    )
    meta = run_llm_compressor_oneshot(lc_plan, output_dir)
    sidecar = write_hygon_infer_sidecar(
        output_dir,
        infer_quantization=hygon_spec.infer_quantization,
        note=hygon_spec.note,
    )
    meta = {
        **meta,
        "backend": "hygon",
        "hygon_infer": str(sidecar),
        "infer_quantization": hygon_spec.infer_quantization,
    }
    return meta


class HygonBackend(QuantBackend):
    """海光 DCU：llm-compressor 压缩 + enginex-hygon-vllm（slimquant / blockwise / awq_marlin）。"""

    name = "hygon"

    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        logger.info(
            "hygon quantize output_dir=%s plan=%s",
            output_dir,
            plan.model_dump_json(indent=2),
        )
        run_pre_oneshot(plan)
        out = Path(output_dir)
        meta = run_hygon_quant(plan, out)
        artifact = run_with_hal(plan, out, meta)
        run_post_oneshot(plan, artifact.output_dir)
        return artifact
