#!/usr/bin/env bash
# Install local llm-compressor into the active venv.
# On Intel Mac, torch>=2.10 wheels do not exist — setup.py auto-relaxes deps;
# real oneshot still needs Linux/CUDA (or Apple Silicon) with torch>=2.10.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/llm-compressor"
INDEX="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"

if [[ ! -d "${SRC}" ]]; then
  echo "missing ${SRC}" >&2
  exit 1
fi

python - <<'PY'
import platform, sys
print(f"python={sys.version.split()[0]} platform={sys.platform} machine={platform.machine()}")
PY

# Optional: install newest torch the index can provide for this platform first.
pip install "torch" -i "${INDEX}" || true

# compressed-tensors also requires torch>=2.10; on Intel Mac install without deps.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
  echo "Intel Mac: installing compressed-tensors --no-deps (oneshot needs Linux/torch>=2.10)"
  pip install "compressed-tensors>=0.18.0" --no-deps -i "${INDEX}" || true
fi

pip install -e "${SRC}" -i "${INDEX}"
python - <<'PY'
from luban_sculpt.backends.llm_compressor.runner import is_llmcompressor_available
ok = is_llmcompressor_available()
print(f"llmcompressor oneshot available: {ok}")
if not ok:
    print("Falling back to LUBAN dry-run on this host is expected.")
PY
