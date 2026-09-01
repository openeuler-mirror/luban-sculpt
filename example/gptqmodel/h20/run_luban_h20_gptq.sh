#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LUBAN_NVIDIA_GPU=h20
export LUBAN_DEVICE_NAME="${LUBAN_DEVICE_NAME:-NVIDIA H20}"

OUT="${OUT:-/data/out/h20-qwen-gptq-w4}"

pip install -e "${ROOT}" -q
pip install gptqmodel datasets -q

unset LUBAN_GPTQMODEL_DRY_RUN

luban-sculpt probe --profile nvidia_h20
luban-sculpt compress \
  --profile nvidia_h20 \
  --recipe "${ROOT}/luban_sculpt/recipes/h20_qwen_gptq_w4.yaml" \
  --output "${OUT}"

echo "vLLM: python -c \"from vllm import LLM; LLM('${OUT}', trust_remote_code=True, quantization='gptq')\""
