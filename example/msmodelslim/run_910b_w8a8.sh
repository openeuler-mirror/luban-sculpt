#!/usr/bin/env bash
# Ascend 910B — W8A8 / W4A8; 不支持 FP8 原生
set -euo pipefail
export LUBAN_ASCEND_SOC=910b
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen2.5-7B-Instruct}"
SAVE_PATH="${SAVE_PATH:-/data/out/910b-qwen-w8a8}"
# --device: npu | npu:0,1,2,3 | cpu
DEVICE="${DEVICE:-npu}"
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --device "${DEVICE}" \
  --model_type Qwen2.5-7B-Instruct \
  --quant_type w8a8 \
  --trust_remote_code True
