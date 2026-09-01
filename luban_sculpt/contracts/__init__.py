"""Layer contracts / IR: shared payloads between HAE, Compiler, Gate, Backend, HAL.

Prefer::

    from luban_sculpt.contracts import BackendPlan, HwDecision, QuantIntent

Implementation lives in ``contracts.ir``.
"""

from luban_sculpt.contracts.ir import (
    ArtifactManifest,
    BackendPlan,
    CalibFusion,
    Fp8Encoding,
    HardwareCapability,
    HwDecision,
    ModelArch,
    ModelArchSnapshot,
    ParallelTopology,
    ProbeResult,
    QuantIntent,
    QuantizedArtifact,
    WeightLayout,
    ExportFormat,
)

__all__ = [
    "ArtifactManifest",
    "BackendPlan",
    "CalibFusion",
    "Fp8Encoding",
    "HardwareCapability",
    "HwDecision",
    "ModelArch",
    "ModelArchSnapshot",
    "ParallelTopology",
    "ProbeResult",
    "QuantIntent",
    "QuantizedArtifact",
    "WeightLayout",
    "ExportFormat",
]
