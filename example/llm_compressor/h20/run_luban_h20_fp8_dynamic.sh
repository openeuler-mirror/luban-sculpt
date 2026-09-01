#!/usr/bin/env bash
# luban-sculpt 编排：H20 + llm-compressor FP8_DYNAMIC
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LUBAN_NVIDIA_GPU="${LUBAN_NVIDIA_GPU:-h20}"
export LUBAN_DEVICE_NAME="${LUBAN_DEVICE_NAME:-NVIDIA H20}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT:-/data/out/h20-qwen2.5-7b-fp8-dynamic}"

pip install -e "${ROOT}" -q
pip install llmcompressor transformers accelerate -q

# 无 GPU 时可 dry-run： export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
unset LUBAN_LLM_COMPRESSOR_DRY_RUN

luban-sculpt probe --profile nvidia_h20
luban-sculpt compress \
  --profile nvidia_h20 \
  --recipe "${ROOT}/luban_sculpt/recipes/h20_qwen_fp8_dynamic.yaml" \
  --output "${OUT}"

echo "manifest:"
cat "${OUT}/manifest.json"
echo ""
echo "vLLM 推理示例:"
echo "  python -c \"from vllm import LLM; LLM('${OUT}', trust_remote_code=True)\""
