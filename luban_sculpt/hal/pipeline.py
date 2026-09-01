"""HAL: calib kernel, encoding, layout, distributed calib."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from luban_sculpt.contracts import CalibFusion, HwDecision, WeightLayout

CalibFn = Callable[..., Any]


@dataclass
class HALPipeline:
    """硬件抽象层：校准 kernel 选择、编码/布局 repack（当前多为 stub，日志进 manifest）。"""

    hw: HwDecision
    repack_log: list[str] = field(default_factory=list)

    def select_calib_kernel(self) -> str:
        """按 calib_fusion 返回 NPU/CUDA 融合或 generic fake-quant 内核名。"""
        if self.hw.calib_fusion == CalibFusion.CUBE_FUSED_MATMUL_SCALE:
            return "npu_fused_matmul_scale"
        if self.hw.calib_fusion == CalibFusion.SIMT_SPLIT_EPILOGUE:
            return "hipblas_gemm_epilogue"
        return "generic_fake_quant"

    def calib_forward(self, layer_name: str, tensor: Any) -> Any:
        """校准前向占位；真实实现应 dispatch 到 select_calib_kernel。"""
        kernel = self.select_calib_kernel()
        # Stub: dispatch to vendor ops
        return {"kernel": kernel, "layer": layer_name, "tensor_ref": str(type(tensor))}

    def encoding_adaptor(self, weights: dict[str, Any]) -> dict[str, Any]:
        """按 fp8_encoding 标记权重编码路径并记录 repack_log。"""
        enc = self.hw.fp8_encoding.value
        self.repack_log.append(f"encoding:{enc}")
        return {**weights, "_encoding": enc}

    def layout_transform(self, weights: dict[str, Any]) -> dict[str, Any]:
        """按 weight_layout 做 NZ/对齐/CT pack 变换（stub 仅打标）。"""
        layout = self.hw.weight_layout
        if layout == WeightLayout.FRACTAL_NZ:
            self.repack_log.append("layout:fractal_nz")
            return {**weights, "_layout": "fractal_nz"}
        if layout == WeightLayout.ROW_MAJOR_ALIGN:
            self.repack_log.append("layout:row_major_align_256")
            return {**weights, "_layout": "row_major_align_256"}
        self.repack_log.append("layout:ct_pack")
        return weights

    def plan_distributed_calib(self, world_size: int) -> dict[str, Any]:
        """返回分布式校准拓扑与 TP/通信后端配置。"""
        return {
            "topology": self.hw.parallel_topology.value,
            "tp_split": self.hw.tp_split_hint,
            "comm": self.hw.comm_backend,
            "world_size": world_size,
        }

    def repack_for_save(self, weights: dict[str, Any]) -> dict[str, Any]:
        """落盘前串联 encoding → layout。"""
        w = self.encoding_adaptor(weights)
        return self.layout_transform(w)
