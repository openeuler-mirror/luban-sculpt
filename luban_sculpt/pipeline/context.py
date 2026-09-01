"""Pipeline runtime context shared across stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luban_sculpt.contracts import (
    ArtifactManifest,
    BackendPlan,
    HwDecision,
    ProbeResult,
    QuantizedArtifact,
)
from luban_sculpt.pipeline.config import PipelineConfig, QuantStageConfig


@dataclass
class StageResult:
    """单阶段执行结果。"""

    stage: QuantStageConfig
    plan: BackendPlan
    artifact: QuantizedArtifact
    output_dir: Path


@dataclass
class PipelineContext:
    """流水线共享上下文：硬件决策、recipe、各阶段产物。"""

    recipe: dict[str, Any]
    recipe_path: Path | None
    output_dir: Path
    profile_name: str
    pipeline: PipelineConfig

    hw: HwDecision | None = None
    probe: ProbeResult | None = None
    profile: dict[str, Any] = field(default_factory=dict)

    stage_results: list[StageResult] = field(default_factory=list)

    @property
    def last_artifact(self) -> QuantizedArtifact | None:
        return self.stage_results[-1].artifact if self.stage_results else None

    @property
    def last_manifest(self) -> ArtifactManifest | None:
        art = self.last_artifact
        return art.manifest if art else None
