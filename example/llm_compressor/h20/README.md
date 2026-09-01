# NVIDIA H20 + llm-compressor 量化示例

H20 为 **Hopper** 架构（compute capability **9.0**），支持 llm-compressor 的 **FP8** 权值/激活量化，导出 **compressed-tensors** 后在 **vLLM (CUDA)** 推理。

## 环境

```bash
# H20 节点
export CUDA_VISIBLE_DEVICES=0
export LUBAN_NVIDIA_GPU=h20   # luban-sculpt probe --profile auto 时选 nvidia_h20

pip install -e /path/to/luban_sculpt
pip install -e ".[llmcompressor]"   # llmcompressor + transformers
pip install vllm                    # 推理阶段
```

| 项 | 建议 |
|----|------|
| 方案 | `FP8_DYNAMIC`（PTQ，激活动态 FP8）或 `FP8_BLOCK`（block 128） |
| Profile | `luban_sculpt/profiles/nvidia_h20.yaml` |
| Recipe | `h20_qwen_fp8_dynamic.yaml` / `h20_qwen_fp8_block.yaml` / `h20_llama3_8b_fp8_dynamic.yaml` |
| 推理 | vLLM 读取 CT 格式，`quant_method: compressed-tensors` |

## 方式 1：luban-sculpt 编排（含 HAL Modifier 链）

```bash
bash example/llm_compressor/h20/run_luban_h20_fp8_dynamic.sh
```

或手动：

```bash
export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export OUT=/data/out/h20-qwen-fp8

luban-sculpt probe --profile nvidia_h20
luban-sculpt compress \
  --profile nvidia_h20 \
  --recipe luban_sculpt/recipes/h20_qwen_fp8_dynamic.yaml \
  --output "${OUT}"
```

无 GPU 开发机可先 dry-run：

```bash
export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
luban-sculpt compress --profile nvidia_h20 \
  --recipe luban_sculpt/recipes/h20_qwen_fp8_dynamic.yaml \
  --output /tmp/h20-dry
```

## 方式 2：纯 llm-compressor 脚本（H20 上直跑）

```bash
python example/llm_compressor/h20/quantize_h20_fp8_dynamic.py
```

环境变量：

- `MODEL_ID`（默认 `Qwen/Qwen2.5-7B-Instruct`）
- `SAVE_DIR`（默认 `./Qwen2.5-7B-Instruct-FP8-Dynamic`）

## vLLM 推理

```python
from vllm import LLM

llm = LLM(
    "/data/out/h20-qwen-fp8",
    trust_remote_code=True,
    max_model_len=8192,
)
print(llm.generate("Hello, my name is"))
```

若 manifest 由 luban-sculpt 生成，其中 `vllm_launch.quant_method` 为 `compressed-tensors`。

## 多卡 H20（可选）

Recipe 中打开分布式校准占位：

```yaml
calib:
  distributed: true
  world_size: 8
```

大模型可参考 llm-compressor [DDP / disk offloading 示例](https://github.com/vllm-project/llm-compressor/tree/main/examples)。

## FP8_BLOCK 变体

```bash
luban-sculpt compress --profile nvidia_h20 \
  --recipe luban_sculpt/recipes/h20_qwen_fp8_block.yaml \
  --output /data/out/h20-qwen-fp8-block
```

## Meta-Llama-3-8B-Instruct（ModelScope）

ModelScope 模型 ID：`LLM-Research/Meta-Llama-3-8B-Instruct`（Hopper H20 → FP8_DYNAMIC）。

```bash
bash example/llm_compressor/h20/run_luban_h20_llama3_fp8_dynamic.sh
# 或覆盖本地路径 / 镜像：
#   MODEL_PATH=/data/models/Meta-Llama-3-8B-Instruct OUT=/data/out/llama3-fp8 \
#     bash example/llm_compressor/h20/run_luban_h20_llama3_fp8_dynamic.sh
```

```bash
luban-sculpt compress --profile nvidia_h20 \
  --recipe luban_sculpt/recipes/h20_llama3_8b_fp8_dynamic.yaml \
  --output /data/out/h20-llama3-8b-fp8-dynamic
```
