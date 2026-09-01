"""Layer contracts / intermediate representation (IR).

Pipeline payloads exchanged between HAE, Compiler, Gate, Backend, HAL, Export.
Not ML model weights — structured DTOs validated by Pydantic.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Model arch types live in luban_sculpt.model (re-exported for convenience)
from luban_sculpt.model.types import ModelArch, ModelArchSnapshot

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


class ExportFormat(str, Enum):
    """量化产物的导出格式（压缩侧写出约定 = vLLM/HF 侧加载协议）。"""

    COMPRESSED_TENSORS = "compressed-tensors"
    AWQ_HF = "awq_hf"
    GPTQ_HF = "gptq_hf"
    FP8_HF = "fp8_hf"
    VLLM_ASCEND = "vllm_ascend"
    INFERA_V1 = "infera-v1"


class CalibFusion(str, Enum):
    CUBE_FUSED_MATMUL_SCALE = "cube_fused_matmul_scale"
    SIMT_SPLIT_EPILOGUE = "simt_split_epilogue"
    GENERIC = "generic"


class Fp8Encoding(str, Enum):
    ASCEND_BLOCK_BF16_PATH = "ascend_block_bf16_path"
    HYGON_UE8M0 = "hygon_ue8m0"
    IEEE_E4M3_REF = "ieee_e4m3_ref"


class WeightLayout(str, Enum):
    FRACTAL_NZ = "fractal_nz"
    ROW_MAJOR_ALIGN = "row_major_align_256"
    CT_PACK = "ct_pack"


class ParallelTopology(str, Enum):
    HCCS_RING = "hccs_ring"
    HCCS_MESH = "hccs_mesh"
    PCIE_SWITCH = "pcie_switch"
    UNKNOWN = "unknown"


class HardwareCapability(BaseModel):
    """HAE detect 后的硬件能力快照：设备身份、算力档位与性能参数。"""

    device_name: str = "unknown"
    vendor: str = "unknown"
    driver_version: str | None = None
    compute_capability: str | None = None
    memory_bytes: int | None = None
    sm_count: int | None = None
    fp8_native: bool = False
    memory_bandwidth_gbps: float | None = None
    peak_tflops_fp16: float | None = None
    stack: dict[str, str] = Field(default_factory=dict)
    source: str = "env"  # api | smi | env


class HwDecision(BaseModel):
    """HAE 输出：校准融合方式、FP8 编码、权重布局、拓扑等，供 HAL / Gate / manifest 使用。"""

    profile_id: str
    vendor: str = "unknown"
    calib_fusion: CalibFusion = CalibFusion.GENERIC
    fp8_encoding: Fp8Encoding = Fp8Encoding.IEEE_E4M3_REF
    weight_layout: WeightLayout = WeightLayout.CT_PACK
    parallel_topology: ParallelTopology = ParallelTopology.UNKNOWN
    tp_split_hint: str = "column"
    comm_backend: str = "nccl"
    stack_snapshot: dict[str, str] = Field(default_factory=dict)
    ascend_soc: str | None = None
    probe_passed_ops: list[str] = Field(default_factory=list)
    probe_failed_ops: list[str] = Field(default_factory=list)


class ProbeResult(BaseModel):
    """能力探测结果：是否通过、日志与缺失算子/栈版本；可选 benchmark 与降级标记。"""

    ok: bool
    messages: list[str] = Field(default_factory=list)
    missing_ops: list[str] = Field(default_factory=list)
    degraded: bool = False
    benchmark: dict[str, Any] = Field(default_factory=dict)


class QuantIntent(BaseModel):
    """Recipe 编译结果：量化意图（算法、部署目标、校准、backend 参数）。"""

    model_id: str
    backend: str
    abstract_scheme: str
    deploy_target: str
    arch_snapshot: ModelArchSnapshot = Field(default_factory=ModelArchSnapshot)
    ignore: list[str] = Field(default_factory=list)
    calib: dict[str, Any] = Field(default_factory=dict)
    backend_options: dict[str, Any] = Field(default_factory=dict)

    @property
    def model_arch(self) -> ModelArch:
        return self.arch_snapshot.arch


class BackendPlan(BaseModel):
    """编排中枢结构：Intent + 硬件决策 + 导出格式。"""

    intent: QuantIntent
    hw: HwDecision
    export_format: ExportFormat


class ArtifactManifest(BaseModel):
    """输出目录 manifest.json：producer、hw、vllm_launch 对账信息。"""

    profile_id: str
    hw_decision: HwDecision
    backend: str
    abstract_scheme: str
    export_format: ExportFormat
    model_arch: str | None = None
    model_arch_snapshot: dict[str, Any] = Field(default_factory=dict)
    tool_version: str = "0.1.0"
    producer: dict[str, Any] = Field(default_factory=lambda: {"name": "luban-sculpt"})
    repack_applied: list[str] = Field(default_factory=list)
    vllm_launch: dict[str, Any] = Field(default_factory=dict)

    def write_json(self, out_dir: Path) -> Path:
        """将 manifest 写入 ``out_dir/manifest.json``。"""
        import json

        path = out_dir / "manifest.json"
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


class QuantizedArtifact(BaseModel):
    """Backend 量化完成后的输出目录与 manifest 句柄。"""

    model_config = {"arbitrary_types_allowed": True}

    output_dir: Path
    manifest: ArtifactManifest
