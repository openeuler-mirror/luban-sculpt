#!/usr/bin/env bash
# msModelSlim V0 传统量化（模型不支持 V1 一键量化时使用）
# 在 msmodelslim 源码根目录执行，参见 example/Qwen/quant_qwen.py
set -euo pipefail
MSM_ROOT="${MSM_ROOT:-/path/to/msmodelslim}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-7B-Instruct}"
SAVE_PATH="${SAVE_PATH:-/data/out/traditional-w8a8}"
cd "${MSM_ROOT}"
python3 example/Qwen/quant_qwen.py \
  --model_path "${MODEL_PATH}" \
  --save_directory "${SAVE_PATH}" \
  --calib_file example/common/boolq.jsonl \
  --w_bit 8 \
  --a_bit 8 \
  --device_type npu \
  --trust_remote_code True
