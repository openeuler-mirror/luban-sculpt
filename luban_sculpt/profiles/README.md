# profiles/ — 芯片能力模板（Hardware Capability Profile）

本目录每个 `*.yaml` 描述 **一块目标芯片/平台** 上，量化与推理该怎么走。

当前保留的 Profile（按手头硬件）：

| id | 平台 |
|----|------|
| `generic_cpu` | CPU / dry-run 默认回退 |
| `hygon_dcu` | 海光 DCU |
| `nvidia_h20` | NVIDIA H20 |
| `ascend_910b` | 昇腾 910B |

与 `recipes/` 的分工：

| | recipes | profiles（本目录） |
|--|---------|-------------------|
| 回答 | 量化什么模型、用什么方案 | 打到哪块卡、编码/布局/拓扑与可用 scheme |
| 示例 | `ascend_qwen_w8a8.yaml` | `ascend_910b.yaml` |
| CLI | `--recipe` | `--profile` |

## 文件里有什么

- **身份**：`id` / `vendor` / `topology_default` / `match_keys`；Ascend 另加 `soc_key`（及可选 `display_name`）
- **能力**：`capability`（`fp8_native`、带宽、峰值算力）— HAE detect 与 Hardware Gate 的 FP8 依据
- **msModelSlim（Ascend）**：`msmodelslim.allowed_quant_types` 等
- **探测**：`required_ops_probe`、非空时的 `stack_gates`
- **HAL 默认**：`hw_defaults`
- **编排**：`backends`、`schemes`（compress backend / export，以及 infer 的 `deploy_target` 或 `runtime`）

本目录只放 YAML 与说明，不含 Python 代码。
