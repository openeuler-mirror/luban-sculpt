"""Validation package: quantized-model checks + pre-compress capability.

Axes
----
- **hf_config** (``validate.hf_config``): 标准 HF ``config.json``
- **runtime** (``validate.runtime``): vLLM import / load / smoke generate
- **quantized_model** (``validate.quantized_model``): 编排上述两项
- **quant_capability** (``validate.quant_capability``): *compress 前* Intent vs profile/芯片

入口（产物）：``validate_quantized_model`` / ``QuantizedModelValidateError``。
入口（压缩前）：``validate_quant_capability`` / ``QuantCapabilityError``。
"""

from luban_sculpt.validate.hf_config import (
    plan_from_hf_config,
    plan_from_manifest,
    validate_compressed_tensors_config,
    validate_fp8_hf_config,
    validate_hf_config,
)
from luban_sculpt.validate.quant_capability import (
    QuantCapabilityError,
    validate_intent_profile,
    validate_quant_capability,
)
from luban_sculpt.validate.quantized_model import (
    QuantizedModelValidateError,
    validate_quantized_model,
    write_quantized_model_report,
)
from luban_sculpt.validate.runtime import RuntimeMode, validate_runtime

__all__ = [
    "QuantCapabilityError",
    "QuantizedModelValidateError",
    "RuntimeMode",
    "plan_from_hf_config",
    "plan_from_manifest",
    "validate_compressed_tensors_config",
    "validate_fp8_hf_config",
    "validate_hf_config",
    "validate_intent_profile",
    "validate_quant_capability",
    "validate_quantized_model",
    "validate_runtime",
    "write_quantized_model_report",
]
