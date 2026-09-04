# llm-compressor Backend + Modifier 拦截

基于 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) 的 `oneshot` API，产出 **compressed-tensors** 格式供 vLLM 加载。

模型侧可设 `model_arch:`（如 `llama` / `qwen`），编译期合并 `ArchQuantPolicy.default_ignore`；见 [详细设计文档 §6](../../docs/详细设计文档.md)。

## 调用链

```
Recipe YAML
  → build_base_modifiers()     # QuantizationModifier(scheme=FP8_DYNAMIC, ...)
  → ModifierManager.apply()    # 国产 Modifier：prepend / append / wrap / replace
  → oneshot(model, recipe)
  → save_pretrained + manifest
```

核心实现见 `luban_sculpt/modifiers/`（`ModifierManager` + `ChainModifier`）。
本 backend 目录仅保留 oneshot runner / scheme_map / LC import。

## Recipe 示例

```yaml
quant:
  backend: llm_compressor
  abstract_scheme: fp8_dynamic
  llm_compressor:
    scheme: FP8_DYNAMIC
    # oneshot(pipeline=...)：顺序加载、逐层量化（降峰值内存）
    # GPTQ 未写时默认 sequential；也可写在 oneshot.pipeline
    pipeline: sequential
    modifiers:
      - name: HALCalibHook
        mode: prepend
      - name: DomesticFakeQuant
        mode: append
```

## 内置 Modifier

| name | 作用 |
|------|------|
| `HALCalibHook` | 按 Profile/HwDecision 选择 HAL 校准 kernel（不改 LC 链） |
| `DomesticFakeQuant` | 插入或占位 `QuantizationModifier` |
| `AscendFp8Block` | `FP8_BLOCK`（x86/实验，非 Ascend 910B 主路径） |

扩展：在自己的包里实现 Modifier，并在 `pyproject.toml` 注册  
`[project.entry-points."luban_sculpt.modifiers"]`（由 `modifier_registry.discover_modifier_classes` 加载）。

## 本机 dry-run

```bash
pip install -e .
export LUBAN_LLM_COMPRESSOR_DRY_RUN=1
bash example/cpu/run_cpu_dry_run.sh
# 或：bash example/llm_compressor/run_llama_fp8_dynamic.sh
```

CPU 专用说明与 pytest 用例见 **[example/cpu/README.md](../cpu/README.md)**。

生成 `llm_compressor_oneshot.py`、`recipe_stub.json`、`luban_oneshot.json`。

## 实机量化

```bash
pip install llmcompressor transformers
unset LUBAN_LLM_COMPRESSOR_DRY_RUN
luban-sculpt compress --profile generic_cpu \
  --recipe luban_sculpt/recipes/llama_fp8_dynamic.yaml \
  --output ./out-fp8-dynamic
```

### NVIDIA H20（Hopper FP8）

见 **[example/llm_compressor/h20/README.md](h20/README.md)**：`nvidia_h20` profile + `h20_qwen_fp8_dynamic.yaml`。

### NVIDIA H20 + GPTQModel（W4 GPTQ）

见 **[example/gptqmodel/h20/README.md](../gptqmodel/h20/README.md)**：`h20_qwen_gptq_w4.yaml` + `pip install gptqmodel`。

参考官方 Quick Tour：[README](https://github.com/vllm-project/llm-compressor/blob/main/README.md)。
