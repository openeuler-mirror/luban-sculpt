#!/usr/bin/env bash
# generic_cpu + dry-run 冒烟（无 GPU/NPU）
# 覆盖：probe → 单 recipe compress → 多阶段 pipeline
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${OUT:-/tmp/luban_cpu_dry_run}"
PIPE_OUT="${PIPE_OUT:-/tmp/luban_cpu_pipeline_dry_run}"
PROBE_JSON="${PROBE_JSON:-/tmp/luban_cpu_probe.json}"

export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
export LUBAN_GPTQMODEL_DRY_RUN=1

pip install -e "${ROOT}" -q

echo "==> [1/3] probe --profile generic_cpu"
luban-sculpt probe --profile generic_cpu | tee "${PROBE_JSON}"
python3 - <<PY
import json
from pathlib import Path
out = json.loads(Path("${PROBE_JSON}").read_text(encoding="utf-8"))
assert out["profile_id"] == "generic_cpu", out
assert out["probe_ok"] is True, out
assert "fp8_dynamic" in out["profile_keys"], out
assert "w4_gptq" in out["profile_keys"], out
print("probe ok")
PY

echo "==> [2/3] compress dry-run (llama_fp8_dynamic)"
rm -rf "${OUT}"
COMPRESS_JSON="$(mktemp)"
luban-sculpt compress \
  --profile generic_cpu \
  --recipe "${ROOT}/luban_sculpt/recipes/llama_fp8_dynamic.yaml" \
  --output "${OUT}" | tee "${COMPRESS_JSON}"

python3 - <<PY
import json
from pathlib import Path
root = Path("${OUT}")
payload = json.loads(Path("${COMPRESS_JSON}").read_text(encoding="utf-8"))
stage = Path(payload["output"])
assert stage.is_dir(), stage
manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
assert manifest["profile_id"] == "generic_cpu", manifest
assert manifest["backend"] == "llm_compressor", manifest
assert manifest["export_format"] == "compressed-tensors", manifest
assert manifest["abstract_scheme"] == "fp8_dynamic", manifest
assert (root / "pipeline_manifest.json").is_file()
print("single compress dry-run ok:", stage)
print("files:", sorted(p.name for p in stage.iterdir()))
PY

echo "==> [3/3] pipeline dry-run (fp8 → gptq)"
rm -rf "${PIPE_OUT}"
luban-sculpt compress \
  --profile generic_cpu \
  --recipe "${ROOT}/luban_sculpt/recipes/pipeline_llm_compressor_then_gptq.yaml" \
  --output "${PIPE_OUT}" > /dev/null

python3 - <<PY
import json
from pathlib import Path
root = Path("${PIPE_OUT}")
assert (root / "pipeline_manifest.json").is_file(), list(root.iterdir())
assert (root / "stage1_fp8" / "manifest.json").is_file()
assert (root / "stage2_gptq" / "manifest.json").is_file()
m1 = json.loads((root / "stage1_fp8" / "manifest.json").read_text(encoding="utf-8"))
m2 = json.loads((root / "stage2_gptq" / "manifest.json").read_text(encoding="utf-8"))
assert m1["profile_id"] == "generic_cpu"
assert m1["abstract_scheme"] == "fp8_dynamic"
assert m2["profile_id"] == "generic_cpu"
assert m2["abstract_scheme"] == "w4_gptq"
assert m2["backend"] == "gptq"
print("pipeline dry-run ok")
PY

echo "CPU dry-run example passed."
echo "  single:   ${OUT}"
echo "  pipeline: ${PIPE_OUT}"
