# Luban Quants

国产化 LLM 量化 CLI/SDK：在 **硬件 Profile（HAE）** 与 **模型架构（ModelArch）** 两条正交轴上编排 PTQ，经 Backend 落盘后由 HAL 写 manifest，供 vLLM / vLLM-Ascend / vLLM-ROCm 加载。

```
Recipe YAML
    │
    ├─ HAE ──────────► HwDecision（芯片 / 编码 / 布局 / 拓扑）
    ├─ ModelArch ────► ArchQuantPolicy（ignore / MoE / 偏好算法）
    │
    ▼
compile → BackendPlan → Hardware Gate
    │
    ▼
Backend（llm_compressor / msmodelslim / lmslim / gptq / …）
    │
    ▼
HAL + manifest.json（+ 各 backend sidecar）
```

## 环境搭建

要求：**Python ≥ 3.10**。推荐用项目内虚拟环境，勿污染系统 Python。

### 1. 创建并激活虚拟环境

```bash
cd luban_sculpt
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

### 4. 打包（可选）

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

### 5. 清理缓存（可选）

```bash
./tools/clean_pycache.sh
```

## 快速开始

```bash
# 硬件探测
luban-sculpt probe --profile auto

# 查看已注册 backend / 模型架构策略 / Ascend 芯片表
luban-sculpt backends
luban-sculpt model-arches
luban-sculpt ascend-chips

# 量化（无对应工具时多为 dry-run，写脚本 + manifest）
luban-sculpt compress \
  --profile generic_cpu \
  --recipe luban_sculpt/recipes/llama_fp8_dynamic.yaml \
  --output ./out

luban-sculpt validate --model ./out
luban-sculpt report --model ./out
```

## 流水线概念

| 概念 | 职责 | 主要路径 |
|------|------|----------|
| HAE | 硬件探测 → `HwDecision` | `hae/` |
| ModelArch | 架构策略 → ignore / MoE / 偏好算法 | `model/` |
| Pipeline | **可扩展编配**：多 backend × algorithm 串联 | `pipeline/` |
| Gate | Fail-Fast 规格校验 | `validate/` |
| Backend | 执行量化 / dry-run | `backends/` |
| HAL / Export | encoding/layout + `manifest.json` | `hal/` `export/` |

### 多阶段 Backend / 算法编配

编排入口：`luban_sculpt.pipeline.QuantPipeline`。

单阶段 Recipe（原 `quant:`）继续可用。多阶段示例：

```yaml
pipeline:
  stages:
    - name: fp8_prep
      backend: llm_compressor
      abstract_scheme: fp8_dynamic
      output_subdir: stage1_fp8
    - name: gptq_w4
      backend: gptq
      algorithm: gptq          # → backend_options.algo
      abstract_scheme: w4_gptq
      input_from: previous     # 以上一阶段产物为 model_id
      output_subdir: stage2_gptq
```

示例 Recipe：`luban_sculpt/recipes/pipeline_llm_compressor_then_gptq.yaml`。  
产物目录含各阶段 `manifest.json` 与总览 `pipeline_manifest.json`。

自定义阶段可实现 `PipelineStage.run(ctx)` 并传入 `QuantPipeline(stages=[...])`。

硬件与模型架构互不替代：同一 `qwen` 可在 `ascend_910b` 或 `hygon_dcu` 上走不同 backend。

## 模型架构（ModelArch）

不同架构的量化差异（dense vs MoE、gate、vision、MLA）集中在 `luban_sculpt/model/`：

```
model/
  types.py          # ModelArch / ModelClass / ArchQuantPolicy
  resolve.py        # 架构决议入口（MoE 规则优先于 dense）
  presets.py        # 合并策略表
  dense/            # 非 MoE：llama / qwen / multimodal …
  moe/              # MoE：qwen_moe / deepseek_moe / mixtral …
```

**解析优先级**

1. Recipe：`model_arch:`
2. 本地权重目录 `config.json` 的 `model_type` / `architectures`
3. `model_id` 路径启发式

**`ArchQuantPolicy` 字段**

| 字段 | 作用 |
|------|------|
| `default_ignore` | 默认跳过的层（如 `lm_head`、Qwen `mlp.gate`） |
| `quant_hints` | MoE 校准放大、router FP16、Ascend `msmodelslim_model_type` 等 |
| `notes` | 说明（CLI `list_arches`） |

`is_moe` / `is_multimodal` 在 **`ModelArchSnapshot`**（运行时），MoE 集合见 **`MOE_ARCHES`**。

详情与扩展方式见 [docs/详细设计文档.md](docs/详细设计文档.md) §6 ModelArch。

## Backend 与芯片

| Backend | 硬件 | Profile 示例 | Wire / 加载 | 文档 |
|---------|------|-------------|-------------|------|
| `llm_compressor` | NVIDIA / 通用 | `nvidia_h20` | compressed-tensors | [example/llm_compressor](example/llm_compressor/README.md) |
| `gptq` | CUDA | `nvidia_h20` | gptq_hf | [example/gptqmodel/h20](example/gptqmodel/h20/README.md) |
| `msmodelslim` | Ascend | `ascend_910b` 等 | vllm_ascend | [example/msmodelslim](example/msmodelslim/README.md) |
| `lmslim` | 海光 DCU | `hygon_dcu` | awq_hf / gptq_hf | [example/lmslim](example/lmslim/README.md) |
| `awq` / `infera` / `custom` | 占位 / 插件 | — | — | — |

### Ascend（msModelSlim）

```bash
export LUBAN_MSMODELSLIM_DRY_RUN=1
luban-sculpt compress --profile ascend_910b \
  --recipe luban_sculpt/recipes/ascend_qwen_w8a8.yaml --output ./out
```

### 海光 DCU（LMSlim）

Wheel：[Sourcefind lmslim](https://download.sourcefind.cn:65024/4/main/lmslim)（按 DTK / Python / torch 选型）。

```bash
export LUBAN_LMSLIM_DRY_RUN=1
luban-sculpt compress --profile hygon_dcu \
  --recipe luban_sculpt/recipes/hygon_qwen_w4a16_awq.yaml --output ./out
```

## Recipe 一览

| Recipe | Profile | Backend | model_arch |
|--------|---------|---------|------------|
| `llama_fp8_dynamic.yaml` | `generic_cpu` | llm_compressor | llama |
| `h20_qwen_fp8_dynamic.yaml` | `nvidia_h20` | llm_compressor | qwen |
| `h20_qwen_fp8_block.yaml` | `nvidia_h20` | llm_compressor | qwen |
| `h20_qwen_gptq_w4.yaml` | `nvidia_h20` | gptq | qwen |
| `ascend_qwen_w8a8.yaml` | `ascend_910b` | msmodelslim | qwen |
| `hygon_qwen_w4a16_awq.yaml` | `hygon_dcu` | awq | qwen |
| `hygon_qwen_w8a8_gptq.yaml` | `hygon_dcu` | gptq | qwen |
| `moe_int4.yaml` | `generic_cpu` | llm_compressor | qwen_moe |

最小字段示例：

```yaml
model_id: /data/models/Qwen2.5-7B-Instruct
model_arch: qwen          # 可选；不写则自动推断
quant:
  backend: lmslim
  abstract_scheme: hygon_w4a16_awq
  deploy_target: vllm_rocm
  ignore: [lm_head]       # 会与 ArchQuantPolicy.default_ignore 合并
  lmslim:
    algo: awq
    strategy: w4a16
calib:
  source: pileval
  max_samples: 128
```

### 校准数据（`calib`）

所有 backend 共用 `luban_sculpt/calib/`（`CalibRunner` → `datasets.iter_text_samples`）。数据**不**内置在仓库里，由 `calib.source` 决定：

| `source` | 来源 | 说明 |
|----------|------|------|
| `stub` | 代码生成 | `datasets.py` 内存造最多 8 条 `"calib stub sample {i}"`，无磁盘文件；多数 recipe 默认，便于 dry-run |
| HF 数据集 id（如 `pileval`、`wikitext`、`allenai/c4`） | HuggingFace `datasets.load_dataset(source, split=...)` | 需安装 `datasets` 且网络/缓存可用；失败则打日志并**回退 stub** |
| 本地 `.jsonl` 路径 | 见下 | `CalibRunner.resolve_calib_file` 可直接引用；`load()` 目前仍按 HF id 尝试，失败回退 stub。msmodelslim 可设 `quant.msmodelslim.calib_file` / `pass_calib_cli` 把路径传给 CLI |

常用键：`source` / `max_samples` / `split` / `text_column` / `batch_size`。  
`pileval` 是量化圈常用的 Pile 验证子集**短名**（海光 AWQ/GPTQ recipe 示例）；若 HF 上该短名不可用，请改成真实 id（如 `mit-han-lab/pile-val-backup`）或本地文件。  
运行后可能在输出目录写出 `luban_calib.jsonl`（sidecar），便于核对或交给 Ascend 工具链。

## 目录结构

| 路径 | 说明 |
|------|------|
| `luban_sculpt/pipeline/` | 可扩展流水线（多 backend × algorithm） |
| `luban_sculpt/hae/` | Hardware-Aware Engine |
| `luban_sculpt/hal/` | 融合校准 / Encoding / Layout |
| `luban_sculpt/model/` | **ModelArch**：dense / moe 策略与分类 |
| `luban_sculpt/compiler/` | Recipe → BackendPlan |
| `luban_sculpt/calib/` | 共享校准：`CalibRunner` + datasets（stub / HF） |
| `luban_sculpt/backends/` | llm_compressor / msmodelslim / lmslim / gptq / … |
| `luban_sculpt/profiles/` | 芯片能力模板 |
| `luban_sculpt/recipes/` | 量化配方 |
| `luban_sculpt/export/` | manifest |
| `luban_sculpt/validate/` | Gate / 规格校验 |
| `luban_sculpt/cli.py` | CLI 入口 |
| `tests/` | 单测（目录与上述模块对齐，见 `tests/README.md`） |
| `example/` | 各 backend 端到端示例 |
| `docs/` | 设计说明（架构、ModelArch） |
| `infera_plugins/vllm/` | 国产卡 vLLM 插件桩 |

## 更多文档

- [docs/概要设计文档.md](docs/概要设计文档.md) — 定位与总体架构（v1.1）
- [docs/详细设计文档.md](docs/详细设计文档.md) — 模块职责与关键接口（v2.5，含 ModelArch）
- [docs/schedule_2026.md](docs/schedule_2026.md) — 海光量化排期计划表（修订版，对齐代码基线）
- [example/msmodelslim/README.md](example/msmodelslim/README.md)
- [example/lmslim/README.md](example/lmslim/README.md)
- [example/llm_compressor/README.md](example/llm_compressor/README.md)
