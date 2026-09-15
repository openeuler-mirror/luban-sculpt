# templates/

通用量化 **模板**，不绑定具体模型。业务 Recipe 通过 `extends: quant` 引用 `quant.yaml` 并覆盖 `model` / `calib` / `pipeline`。

## 快速引用

```yaml
extends: quant

model:
  path: Qwen/Qwen2.5-7B-Instruct
  arch: qwen
  layout: hub_id

pipeline:
  stages:
    - precision: fp8_block   # 覆盖模板默认 fp8_dynamic

calib:
  source: stub
```

## 文件

| 文件 | 说明 |
|------|------|
| `quant.yaml` | 默认单阶段模板；文件头注释为**完整字段说明**（与代码 `pipeline/config.py`、`calib/runner.py` 对齐） |

## 字段速查

### 顶层

| 键 | 必填 | 说明 |
|----|------|------|
| `extends` | 子 recipe | `quant` → 本目录 `quant.yaml` |
| `model` | 是 | 见下表 |
| `pipeline.stages` | 是 | 至少 1 个 stage |
| `calib` | 否 | 默认 stub；见下表 |

### `model`

| 键 | 说明 |
|----|------|
| `path` / `dir` / `id` | 本地目录或 Hub ID |
| `arch` / `model_arch` | `llama`、`qwen`、`qwen_moe` 等 |
| `layout` | `hf_pretrained`（本地校验）\| `hub_id` |

### `pipeline.stages[]`

| 键 | 默认 | 说明 |
|----|------|------|
| `name` | `stage_{i}` | 阶段名 |
| `backend` | `auto` | 或显式 `llm_compressor` / `msmodelslim` / `gptq` … |
| `precision` | — | `fp8_dynamic`、`w8a8`、`w4a16` …（HAE 映射到 scheme） |
| `abstract_scheme` | — | 直接指定 profile scheme 键 |
| `algorithm` | — | `gptq` / `awq` 等 |
| `ignore` | — | 与 ModelArch 默认 ignore 合并 |
| `input_from` | `recipe` | `previous` = 上一阶段输出目录 |
| `output_subdir` | 自动 | 如 `stage1_fp8` |
| `enabled` | `true` | 关闭阶段时不跑 |
| `backend_options` | `{}` | 如 GPTQ `bits` / `group_size` |
| `compress` | `{}` | modifiers、observer、GPTQ 参数等 |
| `<backend>:` | — | 可选块：`llm_compressor` / `msmodelslim` / `gptq` / `awq` / `hygon` / `custom` |

### `compress` 常用子键

| 键 | 用途 |
|----|------|
| `modifiers` | llm-compressor 修饰链 |
| `observer` | weights / input / output 观测器 |
| `algorithm` / `scheme` | GPTQ W4A16 等（也可用 `--precision w4a16`） |

Profile 侧 `backends.*.defaults`（device、observer 默认、Ascend `quant_type`）**不要**写进模板；见 `profiles/nvidia_h20.yaml`、`ascend_910b.yaml`。

### `calib`

| 键 | 说明 |
|----|------|
| `source` | `stub`、HF 数据集名、`open-perfectblend`、路径 |
| `path` / `dir` | 磁盘校准目录 |
| `max_samples` / `batch_size` / `max_seq_length` / `max_chars` |
| `split` / `text_column` / `shuffle` / `seed` |
| `distributed` / `world_size` | 占位 |

CLI：`--calib-preset standard|fast|stub`，`--pipeline-preset fp8_then_gptq`。

## 示例 Recipe

见 [../recipes/README.md](../recipes/README.md)（`llama3.yaml`、`qwen2_5_7b.yaml`、`llama3_fp8_then_gptq.yaml`）。
