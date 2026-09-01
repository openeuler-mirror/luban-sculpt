#!/usr/bin/env python3
"""Standalone llm-compressor 示例：NVIDIA H20 上 Qwen2.5-7B FP8_DYNAMIC.

与 luban-sculpt 解耦，便于在 H20 节点直接跑通后再接编排。
官方文档: https://github.com/vllm-project/llm-compressor
FP8 需 SM >= 8.9（Hopper H20 满足）。
"""
from __future__ import annotations

import os
import sys

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
SAVE_DIR = os.environ.get("SAVE_DIR", "./Qwen2.5-7B-Instruct-FP8-Dynamic")


def _check_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("pip install torch") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请在 H20 节点执行并设置 CUDA_VISIBLE_DEVICES")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}, capability={cap[0]}.{cap[1]}")
    if cap[0] < 8 or (cap[0] == 8 and cap[1] < 9):
        print("警告: FP8 推荐 compute capability >= 8.9 (Hopper/Ada+)")


def main() -> None:
    _check_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"Loading {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    print("Running oneshot (FP8_DYNAMIC) ...")
    oneshot(model=model, recipe=recipe)

    print(f"Saving to {SAVE_DIR} ...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done. Load in vLLM with compressed-tensors (quant_method in config.json).")


if __name__ == "__main__":
    main()
