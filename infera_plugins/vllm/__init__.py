"""vLLM inference-side hooks for luban-sculpt manifests (copy or symlink into vLLM tree)."""

from infera_plugins.vllm.platforms.domestic import DomesticPlatform
from infera_plugins.vllm.kernels.linear.domestic_fp8 import domestic_fp8_linear
from infera_plugins.vllm.kernels.linear.domestic_int4 import domestic_int4_linear

__all__ = [
    "DomesticPlatform",
    "domestic_fp8_linear",
    "domestic_int4_linear",
]
