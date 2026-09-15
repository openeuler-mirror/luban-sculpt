# CPU + Dry-Run 冒烟（generic_cpu）

本目录是 **无加速卡** 时的官方 dry-run 入口：固定 `generic_cpu`，强制 `LUBAN_*_DRY_RUN=1`，不调用真实量化工具。

| 项 | 值 |
|----|-----|
| Profile | `profiles/generic_cpu.yaml`（默认回退） |
| 单阶段 Recipe | `recipes/llama3.yaml` |
| 多阶段 preset | `recipes/llama3.yaml` + `--pipeline-preset fp8_then_gptq` |
| 多阶段 recipe | `recipes/llama3_fp8_then_gptq.yaml` |
| Dry-run 环境变量 | `LUBAN_LLM_COMPRESSOR_DRY_RUN=1`、`LUBAN_GPTQMODEL_DRY_RUN=1` |

## 一键脚本

```bash
cd luban_sculpt
pip install -e .
bash example/cpu/run_cpu_dry_run.sh
```

步骤：

1. `probe --profile generic_cpu`
2. 单 recipe compress dry-run（`llama3.yaml` + `--model-id`）→ manifest + oneshot stub
3. pipeline dry-run（preset：`fp8_then_gptq`）→ `pipeline_manifest.json` + 两阶段目录
4. 打包两阶段 recipe（`llama3_fp8_then_gptq.yaml`）

输出默认：`/tmp/luban_cpu_dry_run`、`/tmp/luban_cpu_pipeline_dry_run`、`/tmp/luban_cpu_packaged_two_stage`（可用 `OUT` / `PIPE_OUT` / `PACKAGED_OUT` 覆盖）。

全仓库测试命令见 [README.md §4 测试](../../README.md)、[tests/README.md](../../tests/README.md)。

## pytest

```bash
export LUBAN_LLM_COMPRESSOR_DRY_RUN=1 LUBAN_GPTQMODEL_DRY_RUN=1
pytest tests/ example/cpu/test_cpu_dry_run.py -q
```

与脚本同一套 dry-run 断言（probe / compile / compress / pipeline）。

## 环境变量

| 变量 | 作用 |
|------|------|
| `LUBAN_LLM_COMPRESSOR_DRY_RUN=1` | llm_compressor 只写 stub，不跑 oneshot |
| `LUBAN_GPTQMODEL_DRY_RUN=1` | gptq 只写配置 stub |

真机请换硬件 profile（`nvidia_h20` / `ascend_910b` / `hygon_dcu`）并 `unset` 上述变量。
