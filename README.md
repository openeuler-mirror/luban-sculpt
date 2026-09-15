# Luban Sculpt

**Luban Sculpt** 是一套面向国产化与多芯片场景的 **LLM 后训练量化命令行工具**：用 Recipe 描述「量化什么」，用 Profile 描述「在哪块硬件上跑」，一条 `luban-sculpt compress` 完成探测、编译、量化与 manifest 落盘，供 vLLM / vLLM-Ascend / vLLM-ROCm 等加载。

---

## 特性

- **一份 Recipe 走多芯片**：同一模型配方 + 不同 `--profile` / `--precision`，不必按 H20、910B 拆多个 YAML。
- **模板化配置**：`extends: quant` 合并 `templates/quant.yaml`；模型、校准、精度在 `recipes/` 里覆盖即可。
- **智能路由**：阶段上 `backend: auto` + `precision`，由 Profile 解析为具体 backend 与 `abstract_scheme`。
- **多阶段流水线**：例如 FP8 预处理再 GPTQ，可用打包 recipe 或 `--pipeline-preset fp8_then_gptq`。
- **无卡可联调**：设置 dry-run 环境变量后，本机无 GPU/NPU 也能跑通脚本与 manifest 生成。

---

## 安装

**要求：Python ≥ 3.10**

```bash
cd luban-sculpt
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip

pip install -r requirements.txt
pip install -e .
```

可选：按能力安装 backend 依赖（体积较大，可按需注释 `requirements.txt` 或使用 extras）：

```bash
pip install -e ".[dev]"
# pip install -e ".[llmcompressor]" / ".[gptqmodel]" / ".[vllm]"
```

| 组件 | 说明 |
|------|------|
| `requirements.txt` | 运行时 + pytest + 可选 backend 包名 |
| `pyproject.toml` | 包元数据、extras、CLI 入口 `luban-sculpt` |

验证安装：

```bash
luban-sculpt --help
python -c "import luban_sculpt; print('ok')"
```

### 可选 Backend

| 包 / Extra | 用途 |
|------------|------|
| `llmcompressor` | NVIDIA / compressed-tensors |
| `gptqmodel` | GPTQ |
| `msmodelslim` | Ascend（需 CANN / torch-npu） |
| `lmslim` | 海光 DCU（厂商 wheel） |
| `vllm` | 运行时校验（默认未装） |

未安装对应工具时，`compress` 通常进入 **dry-run**（生成脚本与 `manifest.json`）。

**Dry-run 环境变量：**

```bash
export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
export LUBAN_GPTQMODEL_DRY_RUN=1
export LUBAN_MSMODELSLIM_DRY_RUN=1
```

---

## 快速使用

```bash
# 查看硬件与能力
luban-sculpt probe --profile auto
luban-sculpt backends
luban-sculpt model-arches

# 同一份 Qwen recipe，换 profile / 精度
luban-sculpt compress \
  --profile nvidia_h20 \
  --recipe luban_sculpt/recipes/qwen2_5_7b.yaml \
  --precision fp8_dynamic \
  --output ./out-h20

luban-sculpt compress \
  --profile ascend_910b \
  --recipe luban_sculpt/recipes/qwen2_5_7b.yaml \
  --precision w8a8 \
  --output ./out-910b

# 本地 Llama 目录
luban-sculpt compress \
  --profile generic_cpu \
  --recipe luban_sculpt/recipes/llama3.yaml \
  --model-dir ./llama3 \
  --output ./out

# 校验与报告（指向阶段输出目录）
luban-sculpt validate --model ./out/stage_0_llm_compressor
luban-sculpt report --model ./out/stage_0_llm_compressor
```

两阶段示例（打包 recipe）：

```bash
luban-sculpt compress \
  --profile generic_cpu \
  --recipe luban_sculpt/recipes/llama3_fp8_then_gptq.yaml \
  --output ./out-two-stage
```

---

## 命令参考

### `compress` 常用参数

| 参数 | 说明 |
|------|------|
| `--recipe` | Recipe YAML 路径（必填） |
| `--output` | 输出根目录（必填） |
| `--profile` | 硬件模板：`auto`、`nvidia_h20`、`ascend_910b`、`generic_cpu` 等 |
| `--precision` | 覆盖首阶段精度，如 `fp8_block`、`w8a8`、`w4a16` |
| `--model-id` / `--model-dir` | 覆盖 Hub ID 或本地权重目录 |
| `--backend` / `--scheme` | 强制 backend 或 scheme（一般保持 recipe 默认 `auto`） |
| `--calib-preset` | `standard` / `fast` / `stub` |
| `--pipeline-preset` | 如 `fp8_then_gptq` 替换整条阶段链 |
| `--ignore` | 逗号分隔，覆盖跳过层列表 |
| `--skip-model-layout-check` | 跳过本地 HF 目录 layout 校验 |

---

## Recipe 与模板

工具读取 YAML Recipe，经 `load_recipe_yaml` 合并模板后执行流水线。

```
templates/quant.yaml  ──extends: quant──►  recipes/*.yaml
        │                                      │
        └──────────── merge ───────────────────┘
                              │
                    luban-sculpt compress
                              │
                    manifest / pipeline_manifest.json
```

### 推荐写法

```yaml
extends: quant

model:
  path: org/model-name-Instruct
  arch: qwen
  layout: hub_id

pipeline:
  stages:
    - backend: auto
      precision: fp8_dynamic
      compress:
        modifiers:
          - name: QuantizationPatch
            mode: patch

calib:
  source: stub
  max_samples: 128
```

- 精度写在 `pipeline.stages[].precision`，或用 CLI `--precision` 覆盖。
- 设备、`trust_remote_code`、Ascend `quant_type` 等写在 **Profile**（`profiles/*.yaml`），不要堆在 Recipe 里。
- 算法相关写在 **`compress:`**（modifiers、observer、GPTQ 参数等）。
- 已废弃顶层 **`quant:`** 块，请勿使用。

### 内置 Recipe（`luban_sculpt/recipes/`）

| 文件 | 说明 |
|------|------|
| `llama3.yaml` | Llama 3；`fp8_dynamic` / `fp8_block` / `w4a16` |
| `qwen2_5_7b.yaml` | Qwen2.5-7B；`fp8_dynamic` / `fp8_block` / `w8a8` |
| `llama3_fp8_then_gptq.yaml` | 两阶段 FP8 → GPTQ W4 |
| `moe_int4.yaml` | MoE 校准样本放大示例 |
| `qwen_observer_smoke.yaml` | Observer 冒烟 |

字段说明：[templates/README.md](luban_sculpt/templates/README.md)、[recipes/README.md](luban_sculpt/recipes/README.md)。

### 校准 `calib`

| `source` | 含义 |
|----------|------|
| `stub` | 内存短样本，适合 dry-run |
| HF 数据集 id | 需 `datasets`，失败回退 stub |
| 本地路径 | 如 `open-perfectblend`（见 `llama3.yaml`） |

---

## 测试

在仓库根目录 `luban-sculpt/`：

```bash
export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
export LUBAN_GPTQMODEL_DRY_RUN=1
export LUBAN_MSMODELSLIM_DRY_RUN=1

pytest tests/ example/cpu/test_cpu_dry_run.py
pytest tests/e2e tests/pipeline/test_two_stage_recipe.py -q
bash example/cpu/run_cpu_dry_run.sh
```

详见 [tests/README.md](tests/README.md)。

---

## 工作原理（简图）

| 模块 | 作用 |
|------|------|
| **HAE + Profile** | `precision` → `abstract_scheme` → backend |
| **ModelArch** | `model.arch` → ignore、MoE 校准策略 |
| **Compiler** | Recipe + Profile → 执行计划 |
| **Pipeline** | 单阶段或多阶段串联各 backend |
| **Backend** | llm_compressor / msmodelslim / gptq / lmslim … |
| **HAL / Export** | 写 `manifest.json` |

多阶段阶段字段示例见 `recipes/llama3_fp8_then_gptq.yaml`。单阶段输出通常在 `stage_0_<backend>/`。

---

## 目录结构

| 路径 | 说明 |
|------|------|
| `luban_sculpt/cli.py` | 命令行入口 |
| `luban_sculpt/templates/` | Recipe 模板 |
| `luban_sculpt/recipes/` | 示例 Recipe |
| `luban_sculpt/profiles/` | 芯片 Profile |
| `luban_sculpt/pipeline/` | 多阶段编排 |
| `luban_sculpt/backends/` | 各量化后端 |
| `luban_sculpt/compiler/` | Recipe 编译 |
| `luban_sculpt/model/` | 模型架构策略 |
| `luban_sculpt/calib/` | 校准数据加载 |
| `tests/` | 单元测试与 dry-run e2e |
| `example/` | 各芯片/后端示例脚本 |
| `docs/` | 设计文档 |

---

## 平台示例

**Ascend（msModelSlim）**

```bash
export LUBAN_MSMODELSLIM_DRY_RUN=1
luban-sculpt compress --profile ascend_910b \
  --recipe luban_sculpt/recipes/qwen2_5_7b.yaml \
  --precision w8a8 --output ./out
```

**海光 DCU（LMSlim）** — 见 [example/lmslim](example/lmslim/README.md)。

**NVIDIA H20（llm-compressor）** — 见 [example/llm_compressor](example/llm_compressor/README.md)。

---

## 更多文档

- [docs/概要设计文档.md](docs/概要设计文档.md)
- [docs/详细设计文档.md](docs/详细设计文档.md)
- [docs/schedule_2026.md](docs/schedule_2026.md)

---

## 打包（可选）

```bash
pip install build wheel
python -m build
# dist/luban_sculpt-*.whl
```

清理 Python 缓存：`./tools/clean_pycache.sh`
