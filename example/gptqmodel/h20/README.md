# GPTQModel + NVIDIA H20 示例

基于 [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) 做 **W4 GPTQ**，权重格式供 **vLLM**（`quantization=gptq`）加载。H20 为 Hopper（SM 9.0），官方矩阵支持 **Marlin / Machete / Exllama** 等 CUDA 内核。

## 环境

```bash
export CUDA_VISIBLE_DEVICES=0
pip install -e /path/to/luban_sculpt
pip install gptqmodel datasets
# 可选加速: pip install gptqmodel[vllm]
```

首次运行会 JIT 编译内核，缓存默认在 `~/.cache/gptqmodel/torch_extensions`。

## 方式 1：luban-sculpt 编排

```bash
bash example/gptqmodel/h20/run_luban_h20_gptq.sh
```

Recipe：`luban_sculpt/recipes/h20_qwen_gptq_w4.yaml`  
Profile：`nvidia_h20`（scheme `w4_gptq` → backend `gptq`）

Dry-run（无 gptqmodel）：

```bash
export LUBAN_GPTQMODEL_DRY_RUN=1
luban-sculpt compress --profile nvidia_h20 \
  --recipe luban_sculpt/recipes/h20_qwen_gptq_w4.yaml \
  --output /tmp/gptq-dry
```

## 方式 2：纯 GPTQModel API

```bash
python example/gptqmodel/h20/quantize_h20_gptq.py
```

环境变量：`MODEL_ID`、`SAVE_DIR`、`CALIB_SAMPLES`、`BATCH_SIZE`。

## vLLM 推理

```python
from vllm import LLM

llm = LLM(
    "/data/out/h20-qwen-gptq-w4",
    trust_remote_code=True,
    quantization="gptq",
)
print(llm.generate("Hello"))
```

## Recipe 字段说明

```yaml
quant:
  backend: gptq
  gptq:
    bits: 4
    group_size: 128
    sym: true
    batch_size: 1      # 按 H20 96GB 显存可调大
    device: cuda:0
calib:
  source: stub         # 或 HuggingFace datasets 路径
  max_samples: 128
```

真实校准可将 `calib.source` 设为 `allenai/c4` 等，并在 `calib` 中配置 `split` / `text_column`（见 `backends/gptq/runner.py`）。
