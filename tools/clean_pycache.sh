#!/usr/bin/env bash
# Remove Python bytecode caches under the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "cleaning __pycache__ / *.pyc / *.pyo under ${ROOT}"

n_dirs=0
n_files=0

while IFS= read -r -d '' dir; do
  rm -rf "${dir}"
  n_dirs=$((n_dirs + 1))
done < <(find "${ROOT}" -type d -name '__pycache__' -print0 2>/dev/null)

while IFS= read -r -d '' file; do
  rm -f "${file}"
  n_files=$((n_files + 1))
done < <(find "${ROOT}" \( -name '*.pyc' -o -name '*.pyo' \) -type f -print0 2>/dev/null)

echo "removed ${n_dirs} __pycache__ dir(s), ${n_files} .pyc/.pyo file(s)"
