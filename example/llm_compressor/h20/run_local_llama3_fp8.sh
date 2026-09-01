#!/usr/bin/env bash
# H20：./llama3 + ./open-perfectblend → FP8_DYNAMIC
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LUBAN_NVIDIA_GPU="${LUBAN_NVIDIA_GPU:-h20}"
export LUBAN_DEVICE_NAME="${LUBAN_DEVICE_NAME:-NVIDIA H20}"
unset LUBAN_LLM_COMPRESSOR_DRY_RUN || true

MODEL_PATH="${MODEL_PATH:-./llama3}"
CALIB_PATH="${CALIB_PATH:-./open-perfectblend}"
OUT="${OUT:-./out-h20-llama3-fp8-dynamic}"
RECIPE_SRC="${ROOT}/luban_sculpt/recipes/h20_llama3_local_fp8_dynamic.yaml"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model dir missing: ${MODEL_PATH}"
  echo "  把 Llama-3 权重放到 ${ROOT}/llama3 （含 config.json / *.safetensors）"
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: ${MODEL_PATH}/config.json not found — 不是 HF 模型目录？"
  exit 1
fi

# 校准：无自定义目录时链到包内 open-perfectblend
if [[ ! -d "${CALIB_PATH}" ]]; then
  PKG_CALIB="${ROOT}/luban_sculpt/calib/open-perfectblend"
  if [[ -d "${PKG_CALIB}" ]]; then
    echo "calib missing at ${CALIB_PATH}; symlink → ${PKG_CALIB}"
    ln -sfn "${PKG_CALIB}" "${CALIB_PATH}"
  else
    echo "ERROR: calib dir missing: ${CALIB_PATH}"
    exit 1
  fi
fi

# 解析为绝对路径，避免 oneshot cwd 变化
MODEL_ABS="$(cd "${MODEL_PATH}" && pwd)"
CALIB_ABS="$(cd "${CALIB_PATH}" && pwd)"
OUT_ABS="$(mkdir -p "${OUT}" && cd "${OUT}" && pwd)"

TMP_RECIPE="$(mktemp -t h20_llama3_local_XXXXXX.yaml)"
trap 'rm -f "${TMP_RECIPE}"' EXIT
sed \
  -e "s|^model_id:.*|model_id: ${MODEL_ABS}|" \
  -e "s|^  path:.*|  path: ${CALIB_ABS}|" \
  "${RECIPE_SRC}" > "${TMP_RECIPE}"

echo "MODEL=${MODEL_ABS}"
echo "CALIB=${CALIB_ABS}"
echo "OUT=${OUT_ABS}"
echo "RECIPE=${TMP_RECIPE}"

pip install -e "${ROOT}" -q
pip install llmcompressor transformers accelerate datasets -q || true

luban-sculpt probe --profile nvidia_h20
luban-sculpt compress \
  --profile nvidia_h20 \
  --recipe "${TMP_RECIPE}" \
  --output "${OUT_ABS}"

echo "done → ${OUT_ABS}"
echo "manifest:"
cat "${OUT_ABS}/manifest.json" 2>/dev/null || true
