# Luban Quants

国产化 LLM 量化 CLI/SDK：在 **硬件 Profile（HAE）** 与 **模型架构（ModelArch）** 两条正交轴上编排 PTQ，经 Backend 落盘后由 HAL 写 manifest，供 vLLM / vLLM-Ascend / vLLM-ROCm 加载。

**设计原则**：同一模型 **一份 Recipe**（不按芯片拆文件）；阶段上 `backend: auto` + `precision`，由 Profile 把精度映射到 `abstract_scheme` 与具体 Backend；工具链参数（device、trust_remote_code、msmodelslim `quant_type` 等）落在 **Profile `backends.*.defaults`**，Recipe 只写 **`compress:`**（modifiers / observer / algorithm）。

```
templates/quant.yaml
        │  extends: quant + model / calib / pipeline 覆盖
        ▼
   Recipe YAML ──► load_recipe_yaml（合并模板）
        │
        ├─ model { path, arch, layout } ──► ModelArch / ignore / MoE 校准放大
        ├─ pipeline.stages[] { backend: auto, precision, compress: … }
        ├─ HAE + Profile ──► precision → abstract_scheme → backend
        │
        ▼
   compile → BackendPlan → Hardware Gate
        │
        ▼
   QuantPipeline（单阶段或多阶段串联）
        │
        ▼
   Backend（llm_compressor / msmodelslim / lmslim / gptq / …）
        │
        ▼
   HAL + manifest.json（+ pipeline_manifest.json）
```

## 环境搭建

要求：**Python ≥ 3.10**。推荐用项目内虚拟环境，勿污染系统 Python。

### 1. 创建并激活虚拟环境

```bash
cd luban-sculpt
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

### 2. 安装依赖

```bash
# 方式 A：requirements（含运行时 + pytest + 可选 backend 包名）
pip install -r requirements.txt
pip install -e .

# 方式 B：按 pyproject extras 精选安装
pip install -e ".[dev]"
# pip install -e ".[llmcompressor]" / ".[gptqmodel]" / ".[vllm]"
```

| 文件 | 对应关系 |
|------|----------|
| `requirements.txt` | 运行时（`pyyaml`/`pydantic`）+ 开发（`pytest`）+ 可选 backend（`llmcompressor`/`gptqmodel`/…） |
| `pyproject.toml` | 正式包元数据、extras 与 entry point（`luban-sculpt` CLI） |

> `llmcompressor` / `gptqmodel` / `msmodelslim` / `transformers` / `datasets` 体积大且常绑 GPU/NPU；本机无对应栈时可先注释 `requirements.txt` 中对应行，或改用 extras 按需安装。`vllm` 默认注释。Ascend 上装 `msmodelslim` 前建议先备好 CANN + torch-npu。

验证：

```bash
pip check
luban-sculpt --help
python -c "import luban_sculpt; print('ok')"
```

### 3. 可选 Backend 说明

| Extra / 包 | 用途 | 备注 |
|------------|------|------|
| `llmcompressor` | NVIDIA / 通用 compressed-tensors | 含 `transformers` |
| `gptqmodel` | GPTQModel | 含 `datasets`（校准 HF 数据集） |
| `vllm` | 运行时校验 / serve | 需本机 GPU 栈匹配；`requirements.txt` 中默认注释 |
| `msmodelslim` | Ascend 量化 | PyPI 包 `msmodelslim`；需 CANN / torch-npu，并保证 CLI 在 `PATH` |
| 海光 lmslim | DCU | 需另装 Sourcefind 等厂商 wheel |

无对应工具时，`compress` 多为 **dry-run**（写脚本 + `manifest.json`），不阻塞本地联调。

常用 dry-run 环境变量：`LUBAN_LLM_COMPRESSOR_DRY_RUN=1`、`LUBAN_GPTQMODEL_DRY_RUN=1`、`LUBAN_MSMODELSLIM_DRY_RUN=1`。

### 4. 测试

在仓库根目录 `luban-sculpt/` 下执行（建议先 `pip install -e .`）：

```bash
cd luban-sculpt

export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
export LUBAN_GPTQMODEL_DRY_RUN=1
export LUBAN_MSMODELSLIM_DRY_RUN=1

# 全量单元测试 + CPU profile 示例测（无 GPU/NPU）
pytest tests/ example/cpu/test_cpu_dry_run.py

# 仅 dry-run 端到端 + 打包两阶段 recipe（llama3_fp8_then_gptq.yaml）
pytest tests/e2e tests/pipeline/test_two_stage_recipe.py -q

# 按模块抽样
pytest tests/calib tests/pipeline tests/compiler tests/hae -q

# CLI 冒烟：probe → 单阶段 llama3 → preset 两阶段 → recipes/llama3_fp8_then_gptq.yaml
bash example/cpu/run_cpu_dry_run.sh
```

e2e 与 `example/cpu/` 用例会设置上述 dry-run 环境变量；本地无加速卡也可跑通。更多目录说明见 [tests/README.md](tests/README.md)。

### 5. 打包（可选）

在已激活的 `.venv` 中：

```bash
pip install build wheel
python -m build
# 产物：dist/luban_sculpt-*.whl 、 dist/luban_sculpt-*.tar.gz
```

其它机器安装 wheel：

```bash
pip install dist/luban_sculpt-0.1.0-py3-none-any.whl
```

### 6. 清理缓存（可选）

```bash
./tools/clean_pycache.sh
```

## 快速开始

```bash
# 硬件探测（profile 决定 vendor、scheme 表、backend 默认参数）
luban-sculpt probe --profile auto

# 查看已注册 backend / 模型架构策略 / Ascend 芯片表
luban-sculpt backends
luban-sculpt model-arches
luban-sculpt ascend-chips

# 单模型、跨芯片：同一份 qwen recipe + 不同 profile / precision
export LUBAN_LLM_COMPRESSOR_DRY_RUN=1   # 无 GPU 时
luban-sculpt compress \
  --profile nvidia_h20 \
  --recipe luban_sculpt/recipes/qwen2_5_7b.yaml \
  --precision fp8_dynamic \
  --output ./out-h20

export LUBAN_MSMODELSLIM_DRY_RUN=1
luban-sculpt compress \
  --profile ascend_910b \
  --recipe luban_sculpt/recipes/qwen2_5_7b.yaml \
  --precision w8a8 \
  --output ./out-910b

# 本地 Llama 权重目录
luban-sculpt compress \
  --profile generic_cpu \
  --recipe luban_sculpt/recipes/llama3.yaml \
  --model-dir ./llama3 \
  --output ./out

luban-sculpt validate --model ./out/stage_0_llm_compressor
luban-sculpt report --model ./out/stage_0_llm_compressor
```

### `compress` 常用 CLI 覆盖

| 参数 | 作用 |
|------|------|
| `--profile` | 芯片能力模板（`auto` / `nvidia_h20` / `ascend_910b` / `generic_cpu` …） |
| `--precision` | 覆盖首阶段 `precision`（如 `fp8_block`、`w8a8`、`w4a16`） |
| `--model-id` / `--model-dir` | 覆盖 `model.path`（Hub ID 或本地 HF 目录） |
| `--backend` / `--scheme` | 强制 backend 或 `abstract_scheme`（一般保持 `auto`） |
| `--calib-preset` | `standard` / `fast` / `stub` 合并校准与 compress 调参 |
| `--pipeline-preset` | 如 `fp8_then_gptq` 替换整条 `pipeline.stages` |
| `--ignore` | 逗号分隔，覆盖 stage `ignore`（与 ModelArch 默认 merge） |
| `--skip-model-layout-check` | 跳过本地 `hf_pretrained` 目录校验 |

## 流水线概念

| 概念 | 职责 | 主要路径 |
|------|------|----------|
| HAE | 硬件探测 + **precision → scheme → backend** | `hae/`、`hae/resolve_quant.py` |
| ModelArch | 架构策略 → ignore / MoE / 偏好算法 | `model/` |
| Pipeline | **可扩展编配**：多 backend × algorithm 串联 | `pipeline/` |
| Gate | Fail-Fast 规格校验 | `validate/` |
| Backend | 执行量化 / dry-run；运行时默认来自 Profile | `backends/`、`backends/runtime_defaults.py` |
| Compiler | `extends` 模板合并 + Recipe → BackendPlan | `compiler/` |
| HAL / Export | encoding/layout + `manifest.json` | `hal/` `export/` |

### 多阶段 Backend / 算法编配

编排入口：`luban_sculpt.pipeline.QuantPipeline`。

Recipe 统一用 `pipeline.stages`（单阶段也写 1 个元素）。阶段可显式写 `backend` / `abstract_scheme`，或使用 **`backend: auto` + `precision`** 由 Profile 解析。

多阶段 YAML 示例（亦见 `recipes/llama3_fp8_then_gptq.yaml`）：

```yaml
pipeline:
  stages:
    - name: fp8_prep
      backend: auto
      precision: fp8_dynamic
      output_subdir: stage1_fp8
    - name: gptq_w4
      backend: gptq
      algorithm: gptq
      abstract_scheme: w4_gptq
      input_from: previous
      output_subdir: stage2_gptq
```

- **打包两阶段**：`recipes/llama3_fp8_then_gptq.yaml`
- **CLI 预设**：`llama3.yaml` + `--pipeline-preset fp8_then_gptq`

产物：各阶段 `manifest.json`、根目录 `pipeline_manifest.json`；单阶段默认在 `stage_0_<backend>/`。

## Recipe 与模板

| 路径 | 说明 |
|------|------|
| `luban_sculpt/templates/quant.yaml` | 通用单阶段模板（勿直接跑，供 `extends`） |
| `luban_sculpt/recipes/*.yaml` | 模型示例与冒烟（见下表） |

新 Recipe 推荐写法：

```yaml
extends: quant

model:
  path: org/model-name-Instruct   # 或本地目录
  arch: qwen                      # llama / qwen / qwen_moe …
  layout: hub_id                  # 本地权重用 hf_pretrained

pipeline:
  stages:
    - backend: auto
      precision: fp8_dynamic      # 可被 --precision 覆盖
      compress:
        observer:                 # 仅算法/观测相关；device 等见 profile
          weights: luban_ema_absmax
          input: minmax

calib:
  source: stub
  max_samples: 128
```

**不再**在 Recipe 顶层写已废弃的 `quant:` 块；精度走 `pipeline.stages[].precision`，方案与 Backend 由 HAE 与 Profile 决议。

### 打包 Recipe 一览

路径均相对于 `luban_sculpt/recipes/`。

| 文件 | 模型 | 精度 / 阶段 | 典型 `--profile` |
|------|------|-------------|------------------|
| `llama3.yaml` | Llama 3 | `fp8_dynamic` / `fp8_block` / `w4a16`（`--precision`） | `nvidia_h20`、`generic_cpu` |
| `qwen2_5_7b.yaml` | Qwen2.5-7B | `fp8_dynamic` / `fp8_block` / `w8a8` | `auto`、`nvidia_h20`、`ascend_910b` |
| `llama3_fp8_then_gptq.yaml` | Llama 3 | 两阶段 FP8 → GPTQ W4 | `generic_cpu`、`nvidia_h20` |
| `moe_int4.yaml` | Qwen MoE | MoE 校准样本放大 | `generic_cpu` |
| `qwen_observer_smoke.yaml` | Qwen 0.5B | observer 冒烟 | `nvidia_h20` |

说明：[recipes/README.md](luban_sculpt/recipes/README.md)、[templates/README.md](luban_sculpt/templates/README.md)。

### 校准数据（`calib`）

所有 backend 共用 `luban_sculpt/calib/`（`CalibRunner` → `datasets.iter_text_samples`）。数据**不**内置在仓库里，由 `calib.source` 决定：

| `source` | 来源 | 说明 |
|----------|------|------|
| `stub` | 代码生成 | 内存短样本；多数模板默认，便于 dry-run |
| HF 数据集 id（如 `pileval`、`wikitext`） | `datasets.load_dataset` | 需 `datasets`；失败回退 stub |
| 本地路径 / `open-perfectblend` 等 | 磁盘 jsonl | Llama 示例 recipe 使用 |

常用键：`source` / `path` / `max_samples` / `batch_size` / `max_seq_length`。运行后可能在输出目录写出 `luban_calib.jsonl`（sidecar）。

## 模型架构（ModelArch）

不同架构的量化差异（dense vs MoE、gate、vision、MLA）集中在 `luban_sculpt/model/`：

```
model/
  types.py          # ModelArch / ArchQuantPolicy
  resolve.py        # 架构决议（MoE 规则优先于 dense）
  dense/            # llama / qwen / multimodal …
  moe/              # qwen_moe / deepseek_moe / mixtral …
```

**解析优先级**

1. Recipe：`model.arch` / `model_arch`
2. 本地权重 `config.json` 的 `model_type` / `architectures`
3. `model.path` / `model_id` 启发式

**`ArchQuantPolicy`**：`default_ignore`（与 stage `ignore` 合并）、`quant_hints`（MoE 校准放大、Ascend `model_type` 等）。

详情见 [docs/详细设计文档.md](docs/详细设计文档.md) §6 ModelArch。

## Backend 与芯片

| Backend | 硬件 | Profile 示例 | Wire / 加载 | 文档 |
|---------|------|-------------|-------------|------|
| `llm_compressor` | NVIDIA / 通用 | `nvidia_h20` | compressed-tensors | [example/llm_compressor](example/llm_compressor/README.md) |
| `gptq` | CUDA | `nvidia_h20` | gptq_hf | [example/gptqmodel/h20](example/gptqmodel/h20/README.md) |
| `msmodelslim` | Ascend | `ascend_910b` 等 | vllm_ascend | [example/msmodelslim](example/msmodelslim/README.md) |
| `lmslim` | 海光 DCU | `hygon_dcu` | awq_hf / gptq_hf | [example/lmslim](example/lmslim/README.md) |
| `awq` / `infera` / `custom` | 占位 / 插件 | — | — | — |

Profile 内 `backends.<name>.defaults` 提供 device、`trust_remote_code`、Ascend `quant_type` 等；Recipe 的 `compress` 只描述 modifiers / observer / GPTQ algorithm 等。

### Ascend（msModelSlim）

```bash
export LUBAN_MSMODELSLIM_DRY_RUN=1
luban-sculpt compress --profile ascend_910b \
  --recipe luban_sculpt/recipes/qwen2_5_7b.yaml \
  --precision w8a8 \
  --output ./out
```

### 海光 DCU（LMSlim）

Wheel：[Sourcefind lmslim](https://download.sourcefind.cn:65024/4/main/lmslim)（按 DTK / Python / torch 选型）。Recipe 与脚本见 [example/lmslim](example/lmslim/README.md)。

```bash
export LUBAN_LMSLIM_DRY_RUN=1
luban-sculpt compress --profile hygon_dcu \
  --recipe <your-recipe.yaml> --output ./out
```

## 目录结构

| 路径 | 说明 |
|------|------|
| `luban_sculpt/templates/` | 通用 Recipe 模板（`quant.yaml` + `extends`） |
| `luban_sculpt/recipes/` | 模型示例与多阶段配方 |
| `luban_sculpt/pipeline/` | QuantPipeline、stage、CLI preset |
| `luban_sculpt/hae/` | Hardware-Aware Engine、精度路由 |
| `luban_sculpt/hal/` | 融合校准 / Encoding / Layout |
| `luban_sculpt/model/` | ModelArch 策略 |
| `luban_sculpt/compiler/` | 模板合并、Recipe → BackendPlan |
| `luban_sculpt/calib/` | CalibRunner + datasets |
| `luban_sculpt/backends/` | 各量化后端实现 |
| `luban_sculpt/profiles/` | 芯片能力模板 |
| `luban_sculpt/export/` | manifest |
| `luban_sculpt/validate/` | Gate / 规格校验 |
| `luban_sculpt/cli.py` | CLI 入口 |
| `tests/` | 单测与 dry-run e2e（**§4 测试**、[tests/README.md](tests/README.md)） |
| `example/` | 各 backend 端到端示例 |
| `docs/` | 设计说明 |
| `infera_plugins/vllm/` | 国产卡 vLLM 插件桩 |

## 更多文档

- [docs/概要设计文档.md](docs/概要设计文档.md) — 定位与总体架构（v1.1）
- [docs/详细设计文档.md](docs/详细设计文档.md) — 模块职责与关键接口（v2.5，含 ModelArch）
- [docs/schedule_2026.md](docs/schedule_2026.md) — 海光量化排期计划表（修订版，对齐代码基线）
- [example/msmodelslim/README.md](example/msmodelslim/README.md)
- [example/lmslim/README.md](example/lmslim/README.md)
- [example/llm_compressor/README.md](example/llm_compressor/README.md)
