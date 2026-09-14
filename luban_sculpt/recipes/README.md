# recipes/

| 文件 | 用途 |
|------|------|
| [`../templates/quant.yaml`](../templates/quant.yaml) | 通用模板（`extends: quant`） |
| `llama3.yaml` | Llama 3：`fp8_dynamic` / `fp8_block` / `w4a16`（`--precision`） |
| `qwen2_5_7b.yaml` | Qwen2.5-7B：`fp8_dynamic` / `fp8_block` / `w8a8`（`--precision`） |
| `llama3_fp8_then_gptq.yaml` | Llama 3 两阶段：FP8 → GPTQ W4 |
| `qwen_observer_smoke.yaml` | Observer 冒烟 |
| `moe_int4.yaml` | MoE 校准样本放大（`qwen_moe`） |

新模型：复制 `llama3.yaml` 或 `qwen2_5_7b.yaml`，改 `model.path` / `model.arch`。
