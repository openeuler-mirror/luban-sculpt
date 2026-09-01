#!/usr/bin/env python3
"""H20 上 GPTQModel 4bit 量化（独立脚本，与 luban-sculpt 解耦）。

参考: https://github.com/ModelCloud/GPTQModel
Hopper (H20) 支持 Marlin / Machete 等 CUDA 内核（Turing+ sm_75+）。
"""
from __future__ import annotations

import os

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
SAVE_DIR = os.environ.get("SAVE_DIR", "./Qwen2.5-7B-Instruct-GPTQ-4bit")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
CALIB_SAMPLES = int(os.environ.get("CALIB_SAMPLES", "128"))


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA（NVIDIA H20）")
    print("Device:", torch.cuda.get_device_name(0))

    from gptqmodel import GPTQConfig, GPTQModel

    # 简单 stub 校准；生产可换 allenai/c4 等
    calibration_dataset = [f"calibration sample {i} for GPTQ." for i in range(CALIB_SAMPLES)]

    quant_config = GPTQConfig(bits=4, group_size=128, sym=True)
    model = GPTQModel.load(MODEL_ID, quant_config)
    model.quantize(calibration_dataset, batch_size=BATCH_SIZE)
    model.save(SAVE_DIR)
    print(f"Saved to {SAVE_DIR}")
    print(f'vLLM: LLM("{SAVE_DIR}", quantization="gptq", trust_remote_code=True)')


if __name__ == "__main__":
    main()
