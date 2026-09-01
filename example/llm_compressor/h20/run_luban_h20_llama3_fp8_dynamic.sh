#!/usr/bin/env bash
# luban-sculpt：H20 + Meta-Llama-3-8B-Instruct FP8_DYNAMIC
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LUBAN_NVIDIA_GPU="${LUBAN_NVIDIA_GPU:-h20}"
export LUBAN_DEVICE_NAME="${LUBAN_DEVICE_NAME:-NVIDIA H20}"

# ModelScope 默认 ID；本地目录可：MODEL_PATH=/data/models/Meta-Llama-3-8B-Instruct
MODEL_PATH="${MODEL_PATH:-LLM-Research/Meta-Llama-3-8B-Instruct}"
OUT="${OUT:-/data/out/h20-llama3-8b-fp8-dynamic}"
RECIPE="${ROOT}/luban_sculpt/recipes/h20_llama3_8b_fp8_dynamic.yaml"

pip install -e "${ROOT}" -q
pip install llmcompressor transformers accelerate modelscope -q

# 无 GPU 时可 dry-run： export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
# unset LUBAN_LLM_COMPRESSOR_DRY_RUN

# 若 MODEL_PATH 与 recipe 内 model_id 不同，写临时 recipe 覆盖
TMP_RECIPE="$(mktemp -t h20_llama3_XXXXXX.yaml)"
trap 'rm -f "${TMP_RECIPE}"' EXIT
sed "s|^model_id:.*|model_id: ${MODEL_PATH}|" "${RECIPE}" > "${TMP_RECIPE}"

luban-sculpt probe --profile nvidia_h20
luban-sculpt compress \
  --profile nvidia_h20 \
  --recipe "${TMP_RECIPE}" \
  --output "${OUT}"

echo "manifest:"
cat "${OUT}/manifest.json"
echo ""
echo "vLLM 推理示例:"
echo "  python -c \"from vllm import LLM; LLM('${OUT}', trust_remote_code=True)\""
