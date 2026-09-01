# msModelSlim 接入示例（Ascend 910B）

本目录演示如何通过 **luban-sculpt** 编排 [MindStudio ModelSlim (msModelSlim)](https://github.com/Ascend/msmodelslim/tree/master) 一键量化。当前仓库仅保留 **Ascend 910B** profile（W8A8 / W4A8，无 FP8 原生）。

官方快速入门见 [Quantization Quickstart](https://github.com/Ascend/msmodelslim/blob/master/docs/en/getting_started/quantization_quick_start.md)。

## 环境准备

1. 安装 CANN + torch-npu，并安装 **msModelSlim**（确保 `msmodelslim` 在 `PATH`）。
2. 安装 luban-sculpt：`pip install -e .`（在 `luban_sculpt/` 根目录）。
3. 指定芯片（可选，默认按 910B）：

```bash
export LUBAN_ASCEND_SOC=910b
export ASCEND_RT_VISIBLE_DEVICES=0
```

## luban-sculpt 编排（推荐）

未安装 msModelSlim 时默认 **dry-run**，会生成 `msmodelslim_command.sh` + `manifest.json`：

```bash
export LUBAN_MSMODELSLIM_DRY_RUN=1

luban-sculpt probe --profile ascend_910b
luban-sculpt compress \
  --profile ascend_910b \
  --recipe luban_sculpt/recipes/ascend_qwen_w8a8.yaml \
  --output /data/out/qwen2.5-7b-w8a8
```

## 本目录脚本

| 脚本 | 说明 |
|------|------|
| `run_910b_w8a8.sh` | 910B W8A8 |
| `run_luban_orchestrated.sh` | luban-sculpt compress 一键 |
| `traditional_qwen.sh` | msModelSlim V0 传统脚本入口说明 |

修改脚本内 `MODEL_PATH` / `SAVE_PATH` 后执行即可。
