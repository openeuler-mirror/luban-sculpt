#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export LUBAN_ASCEND_SOC="${LUBAN_ASCEND_SOC:-910b}"
export LUBAN_MSMODELSLIM_DRY_RUN="${LUBAN_MSMODELSLIM_DRY_RUN:-1}"
OUT="${OUT:-/tmp/luban_msmodelslim_out}"
pip install -e "${ROOT}" -q
luban-sculpt compress \
  --profile "ascend_${LUBAN_ASCEND_SOC}" \
  --recipe "${ROOT}/luban_sculpt/recipes/ascend_qwen_w8a8.yaml" \
  --output "${OUT}"
echo "Generated: ${OUT}/msmodelslim_command.sh"
