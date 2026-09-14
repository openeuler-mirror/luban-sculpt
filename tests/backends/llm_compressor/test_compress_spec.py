"""backends.compress_spec ↔ llm-compressor examples/quantization_*."""

from __future__ import annotations

import pytest

from luban_sculpt.backends.compress_spec import (
    resolve_compress_spec,
    scheme_for_abstract,
)


@pytest.mark.parametrize(
    "abstract,expected_scheme,algo",
    [
        ("w4a16", "W4A16", "gptq"),
        ("w4a16_fp4", "NVFP4A16", "quantization"),
        ("w4a4_fp4", "NVFP4", "quantization"),
        ("nvfp4", "NVFP4", "quantization"),
        ("w4a4_mxfp4", "MXFP4", "quantization"),
        ("w4a8_fp8", "W4AFP8", "gptq"),
        ("w8a8_fp8", "FP8_DYNAMIC", "quantization"),
        ("fp8_dynamic", "FP8_DYNAMIC", "quantization"),
        ("w8a8_int8", "W8A8", "gptq"),
        ("w8a8_mxfp8", "MXFP8", "quantization"),
        ("fp8_block", "FP8_BLOCK", "quantization"),
    ],
)
def test_scheme_and_default_algo(
    abstract: str, expected_scheme: str, algo: str
) -> None:
    assert scheme_for_abstract(abstract) == expected_scheme
    spec = resolve_compress_spec(abstract)
    assert spec.scheme == expected_scheme
    assert spec.algorithm == algo


def test_w4a16_gptq_block_size() -> None:
    spec = resolve_compress_spec("w4a16")
    assert spec.block_size == 128
    assert spec.quantization_format == "pack-quantized"


def test_w8a8_int8_no_default_block_size() -> None:
    spec = resolve_compress_spec("w8a8_int8")
    assert spec.algorithm == "gptq"
    assert spec.block_size is None


def test_unknown_scheme_falls_back_to_fp8_dynamic() -> None:
    assert scheme_for_abstract("not_a_scheme") is None
    spec = resolve_compress_spec("not_a_scheme")
    assert spec.scheme == "FP8_DYNAMIC"
    assert spec.algorithm == "quantization"
