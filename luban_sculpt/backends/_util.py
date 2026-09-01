"""Shared backend helper: HAL + manifest."""

from __future__ import annotations

from pathlib import Path

from luban_sculpt.export.metadata import build_manifest
from luban_sculpt.hal.pipeline import HALPipeline
from luban_sculpt.contracts import BackendPlan, QuantizedArtifact
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


def run_with_hal(plan: BackendPlan, output_dir: Path, weights_stub: dict) -> QuantizedArtifact:
    """dry-run / stub 后端共用：HAL repack + build_manifest + 写 manifest.json。"""
    logger.info(
        "HAL+manifest start backend=%s output=%s",
        plan.intent.backend,
        output_dir,
    )
    try:
        hal = HALPipeline(hw=plan.hw)
        if plan.intent.calib.get("distributed"):
            hal.plan_distributed_calib(int(plan.intent.calib.get("world_size", 1)))
        hal.calib_forward("stub_layer", None)
        repacked = hal.repack_for_save(weights_stub)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "quant_stub.json").write_text(str(repacked), encoding="utf-8")
        manifest = build_manifest(plan, hal.repack_log)
        manifest.write_json(output_dir)
    except Exception:
        logger.error("HAL+manifest failed output=%s", output_dir, exc_info=True)
        raise
    logger.info(
        "HAL+manifest done output=%s export=%s repack=%s",
        output_dir,
        manifest.export_format.value,
        manifest.repack_applied,
    )
    return QuantizedArtifact(output_dir=output_dir, manifest=manifest)
