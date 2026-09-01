# CPU + Dry-Run 冒烟（generic_cpu）

本目录是 **无加速卡** 时的官方 dry-run 入口：固定 `generic_cpu`，强制 `LUBAN_*_DRY_RUN=1`，不调用真实量化工具。

| 项 | 值 |
|----|-----|
| Profile | `profiles/generic_cpu.yaml`（默认回退） |
| 单阶段 Recipe | `recipes/llama_fp8_dynamic.yaml` |
| 多阶段 Recipe | `recipes/pipeline_llm_compressor_then_gptq.yaml` |
| Dry-run 环境变量 | `LUBAN_LLM_COMPRESSOR_DRY_RUN=1`、`LUBAN_GPTQMODEL_DRY_RUN=1` |

## 一键脚本

```bash
cd luban_sculpt
pip install -e .
bash example/cpu/run_cpu_dry_run.sh
```

步骤：

1. `probe --profile generic_cpu`
2. 单 recipe compress dry-run → manifest + oneshot stub
3. pipeline dry-run（FP8 → GPTQ）→ `pipeline_manifest.json` + 两阶段目录

输出默认：`/tmp/luban_cpu_dry_run`、`/tmp/luban_cpu_pipeline_dry_run`（可用 `OUT` / `PIPE_OUT` 覆盖）。

## pytest

```bash
pytest example/cpu/test_cpu_dry_run.py -q
```

与脚本同一套 dry-run 断言（probe / compile / compress / pipeline）。

## 环境变量

| 变量 | 作用 |
|------|------|
| `LUBAN_LLM_COMPRESSOR_DRY_RUN=1` | llm_compressor 只写 stub，不跑 oneshot |
| `LUBAN_GPTQMODEL_DRY_RUN=1` | gptq 只写配置 stub |

真机请换硬件 profile（`nvidia_h20` / `ascend_910b` / `hygon_dcu`）并 `unset` 上述变量。
