# Test layout mirrors ``luban_sculpt/`` modules.

| 目录 | 对应模块 |
|------|----------|
| `tests/calib/` | `luban_sculpt.calib` |
| `tests/pipeline/` | `luban_sculpt.pipeline`（含 `test_two_stage_recipe.py`） |
| `tests/compiler/` | `luban_sculpt.compiler`（含 `extends: quant` 模板合并） |
| `tests/model/` | `luban_sculpt.model` |
| `tests/hae/` | `luban_sculpt.hae` |
| `tests/validate/` | `luban_sculpt.validate` |
| `tests/backends/` | `luban_sculpt.backends.*` |
| `tests/recipes/` | 打包 recipe（Ascend / msmodelslim） |
| `tests/log/` | `luban_sculpt.log` |
| `tests/cli/` | `luban_sculpt.cli` |
| `tests/e2e/` | Dry-run 完整链路（probe → compress → validate → report） |
| `tests/paths.py` | 仓库根 / `recipes/` / `templates/` / profiles 路径 |

## 推荐命令

在 **`luban-sculpt/`** 仓库根目录：

```bash
pip install -e .

export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
export LUBAN_GPTQMODEL_DRY_RUN=1
export LUBAN_MSMODELSLIM_DRY_RUN=1

# 全量（当前 CI 本地等价）
pytest tests/ example/cpu/test_cpu_dry_run.py

# 安静摘要
pytest tests/ example/cpu/test_cpu_dry_run.py -q

# dry-run 端到端 + 两阶段打包 recipe
pytest tests/e2e tests/pipeline/test_two_stage_recipe.py -q

# 按模块
pytest tests/calib tests/pipeline tests/compiler tests/hae -q
pytest tests/recipes tests/model tests/cli -q
```

## Shell 冒烟（无 GPU）

```bash
bash example/cpu/run_cpu_dry_run.sh
```

覆盖：`probe` → `recipes/llama3.yaml` 单阶段 → `llama3.yaml` + `--pipeline-preset fp8_then_gptq` → `recipes/llama3_fp8_then_gptq.yaml`。

## 从 monorepo 上级目录跑

```bash
pytest luban-sculpt/tests/ -q
```
