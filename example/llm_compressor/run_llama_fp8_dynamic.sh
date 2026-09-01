#!/usr/bin/env bash
# 兼容入口：转发到 example/cpu dry-run 冒烟
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "${ROOT}/example/cpu/run_cpu_dry_run.sh"
