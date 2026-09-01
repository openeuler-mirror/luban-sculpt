# 海光量化（luban-sculpt）排期计划表（修订版）

> **基准日期**：2026-08-24  
> **代码基线**：`luban_sculpt/`（HAE × ModelArch × 多 Backend 骨架已落地）  
> **修订原则**：原计划保留里程碑节奏；已完成项收口为文档交付；未完成项按代码缺口重排，优先 **Policy 执行贯通 → 校准 → DeepSeek-MoE 海光通路 → Qwen3 增强**。

---

## 0. 与原计划对照（现状盘点）

| 原计划模块 | 代码现状 | 修订处理 |
|------------|----------|----------|
| 技术选型调研 | 已选型：lmslim / msmodelslim / llm_compressor / gptq | **收口写报告**（8.24–8.25） |
| 概要设计 | `docs/概要设计文档.md` 已有 | **补协议章节**（若需） |
| 详细设计 | `docs/详细设计文档.md`（含 ModelArch）；缺 Pass / 时序 / 指标 | **9.1–9.10 补齐** |
| 海光 Backend 底座 | `backends/lmslim/` + `hygon_dcu` profile **骨架 DONE** | 升级为「算子/内存/通信」真实现 |
| Compiler Pass 调度 | 仅有 Recipe→`BackendPlan`，**无 Pass 图** | 新建 Pass 调度器 |
| 模型导出/验证 | `export/metadata` + CLI validate **PARTIAL** | 补导出校验与配置回写 |
| HAE LUT | `profiles/*.yaml` 已作能力表 | 扩展为可查询性能 LUT API |
| 模型结构 IR | 仅有 `ModelArch`/`ArchQuantPolicy`，**非图 IR** | 定义 Quant IR（Dense/MoE） |
| 校准 KL/MinMax | `calib/` **stub** | V1 从 DeepSeek 阶段启动 |
| DeepSeek MoE | 仅 classify + policy 提示 | **策略执行 + calib + lmslim**，非「专家并行/IR Pass」 |
| Qwen3 GQA/KV/hgemm | **未开始** | 按原窗口 11.3–11.30 |
| 精度门禁 | Gate 仅硬件，无 PPL | 并入 M3 起强制 |

---

## 1. 排期总表（修订）

| 阶段 | 日期 | 类型 | 任务（对齐 luban_sculpt） | 交付物 | 状态 |
|------|------|------|---------------------------|--------|------|
| **调研收尾** | 8.24–8.25 | 调研 | 完成技术选型调研报告：明确以 **luban-sculpt 编排 + lmslim(海光)/msmodelslim(昇腾)/llm_compressor(通用)** 为量化框架；对比 AWQ/GPTQ/W8A8/FP8 适用边界 | 《技术选型分析调研报告》 | 待收口 |
| **概要设计** | 8.25–8.30 | 概要设计 | 固化系统架构图；模块边界：HAE / ModelArch / Compiler / Backend / HAL / Export；定义 **Backend ↔ Compiler 交互协议**（`BackendPlan` / `QuantIntent` / ExportFormat） | 《概要设计文档》（含架构图） | 文档骨架已有，待正式化 |
| **详细设计** | 9.1–9.10 | 详细设计 | 输出时序图（compress 全链路）、异常处理（dry-run / BackendUnavailable / Gate）、性能指标定义；定义 **统一 Quant IR**；选型测试框架（pytest + GPU/CPU 逐位比对策略） | 《详细设计文档》、测试框架选型报告、IR 定义初稿 | 待做 |
| **基础底座** | 9.11–10.5 | Backend | 封装海光 DCU **lmslim** 后端：Ignore Adapter、内存/设备通信骨架、与 Sourcefind wheel 契约固化 | 基础算子/适配层代码、通信接口文档 | 骨架 DONE → 增强 |
| **基础底座** | 9.11–10.5 | Compiler | 完成 **Pass 调度器骨架**，预留量化 Pass 插入点（在现有 `recipe_compiler` 之上） | Compiler 骨架、Pass 接口定义 | 待做 |
| **基础底座** | 9.11–10.5 | 导出模块 | 量化后导出、manifest 校验、HF/`quantization_config` 回写、vLLM 启动提示完整性 | 导出模块测试通过 | PARTIAL → 完成 |
| **基础底座** | 9.11–10.5 | HAE | 硬件性能 LUT 初始化接口（在 `profiles/` 之上提供查询 API：算子/精度/带宽档） | 性能数据接口 + 初始配置 | PARTIAL → 完成 |
| **基础底座** | 9.11–10.5 | 模型结构 | 统一 Quant IR：支持 Dense / MoE 扩展；与现有 `ModelArch`/`ArchQuantPolicy` 映射 | IR 详细文档 + 基础类 | PARTIAL → 完成 |
| **基础底座** | 9.11–10.5 | Policy 贯通 | **P0**：Ignore 方言解析 + lmslim/gptq/msmodelslim Adapter；MoE 不支持路径 fail-fast | 跨 Backend ignore 单测 | 新增（原计划隐含，代码必补） |
| **底座验收 M2** | **10.5** | 集成验证 | 「导入 → IR 转换 → Backend 分配 → dry-run/空转」闭环 | 空转流程测试报告及演示 | 里程碑 |
| **DeepSeek 海光** | 10.6–11.2 | 校准 V1 | 实现 KL / MinMax 静态校准，对接 **lmslim GPTQ 校准样本**（`calib/` 替换 stub）；MoE 校准集覆盖 routed/shared expert 激活 | 校准模块代码 + 单测 | 待做 |
| **DeepSeek 海光** | 10.6–11.2 | **MoE 策略执行** | **`model_arch: deepseek_moe`**：补全 router / `eh_proj` / shared expert 等 **HF 模块名**；`ArchQuantPolicy` → **Ignore 解析** → **lmslim exclude**；`router_fp16` / `quant_hints` **可验收**（非仅 metadata） | DeepSeek MoE 策略单测 + lmslim 集成用例 | 待做 |
| **DeepSeek 海光** | 10.6–11.2 | **HAL 实装（海光）** | `hal/pipeline` **encoding/layout 真实现**（`hygon_ue8m0`、`row_major_align_256`）；校准期 kernel 选择与 `hygon_dcu` profile 一致；manifest/repack 可对账 | HAL 单测 + repack 日志样例 | 待做 |
| **DeepSeek 海光** | 10.6–11.2 | **Policy Pass** | **Pass-Lite**：在 `recipe_compiler` 后执行策略 Pass（ignore 合并、MoE calib 放大、lmslim algo）；**不含** IR 图插 Q/DQ（整模 PTQ 在 lmslim 内完成） | Policy Pass + 集成测试 | 待做 |
| **DeepSeek 海光** | 10.6–11.2 | 精度门禁 | 量化后 **PPL / smoke gen** 阈值（recipe 可配）；挂 `QuantPipeline` 自定义阶段后置 | 精度门禁代码 + M3 报告模板 | 待做 |
| **DeepSeek 海光** | 10.6–11.2 | 通路测试 | DeepSeek 约定规模 **lmslim 实跑**（非逐算子 GPU/CPU 逐位）；dry-run 与 DCU 金机对照 | 通路测试报告 + Bug 记录 | 待做 |
| **DeepSeek M3** | **11.2** | 整体验证 | DeepSeek-7B（或约定规模）海光量化推理，**精度损失 &lt;1%**（PPL/任务指标写入 recipe） | 验证报告 + 可执行模型 | 里程碑 |
| **Qwen3 海光** | 11.3–11.30 | 校准 V2 | Qwen3 注意力动态校准策略 | 校准 V2 + 单测 | 待做 |
| **Qwen3 海光** | 11.3–11.30 | 结构适配 | GQA 结构适配；独立 `qwen3` ModelArch（若需与 qwen2 策略分叉） | 适配代码 + 用例 | 待做 |
| **Qwen3 海光** | 11.3–11.30 | 导出增强 | 海光专用格式导出（**.hgemm** 或项目确认的 DCU 打包格式） | 导出代码 + 格式文档 | 待做 |
| **Qwen3 海光** | 11.3–11.30 | Compiler 优化 | Qwen3 形状优化 + **KV Cache 量化** Pass | 优化 Pass + 性能对比 | 待做 |
| **Qwen3 海光** | 11.3–11.30 | 模测 | 覆盖 1.8B / 7B / 72B（或可用替代规模）单项功能 | 测试报告 | 待做 |
| **Qwen3 M4** | **11.30** | 整体验证 | Qwen3 系列海光量化编译与运行通路（原表「11.3」更正为阶段末） | 通路验证报告 | 里程碑 |
| **集成验证** | 12.1–12.10 | 框架集成 | 端到端编译–执行闭环稳定性 | 集成测试报告 | 待做 |
| **集成验证** | 12.1–12.18 | 结构对比 | Dense / MoE / GQA 量化效果对比 | 量化效果对比报告 | 待做 |
| **集成验证** | 12.10–12.25 | 性能压测 | 吞吐 / 延迟 / 显存边界 + 回归修复 | 性能测试报告 | 待做 |
| **集成验证** | 12.20–12.25 | 文档清理 | 《海光量化调优指南》+ 代码注释整理 | 调优指南、清理记录 | 待做 |
| **终验 M5/M6** | **12.26–12.30** | 全量交付 | 全量回归；源码 / 部署脚本 / Docker / 报告打包；**12.30 终验评审** | 完整交付包 | 里程碑 |

---

## 2. 里程碑（修订）

| 里程碑 | 日期 | 验收标准（可测） |
|--------|------|------------------|
| **M1** 设计冻结 | 9.10 | 《概要设计》《详细设计》、IR 初稿、测试框架选型签字 |
| **M2** 底座空转 | 10.5 | `luban-sculpt compress`：HF 导入 → Quant IR → Backend 分配 → dry-run 产物 + manifest；Ignore 跨 lmslim 生效单测通过 |
| **M3** DeepSeek 海光 | 11.2 | 约定规模 DCU **lmslim 整模量化**可跑通；PPL/任务 **精度损失 &lt;1%**；MoE **exclude/router 策略在 manifest 与产物可对账** |
| **M4** Qwen3 通路 | 11.30 | Qwen3 多尺寸量化+推理通路；GQA/KV 相关 Pass 有性能对比数据 |
| **M5** 集成稳定 | 12.18 | Dense/MoE/GQA 对比报告 + 集成稳定性数据 |
| **M6** 终验 | 12.30 | 压测报告 + 调优指南 + 镜像/脚本/源码全量交付 |

---

## 3. 与代码模块的映射（执行时按此拆任务）

| 计划任务 | 主要落点 |
|----------|----------|
| 技术选型 / 概要设计 | `docs/` + 新增《技术选型》《概要设计》正式稿 |
| Backend↔Compiler 协议 | `contracts/`（`BackendPlan`/`QuantIntent`）、`compiler/`、`backends/base.py` |
| Pass 调度器 | 新建 `compiler/passes/`（在 `recipe_compiler` 之上） |
| Quant IR | 新建 `model/ir/`（Dense/MoE Node）；`ModelArch` 作结构标签 |
| lmslim / 海光 | `backends/lmslim/`、`profiles/hygon_dcu.yaml`、`example/lmslim/` |
| Ignore Adapter（P0） | `model/resolve` 解析 + 各 `backends/*/runner` 消费 |
| HAE LUT | `hae/` + `profiles/` 查询 API |
| 导出/验证 | `export/`、`validate/`、CLI `validate`/`report` |
| 校准 V1/V2 | `calib/`（替换 stub） |
| DeepSeek-MoE | `model/moe/presets.py` + **Ignore Adapter** + `backends/lmslim/` + expert-aware `calib/` |
| HAL 海光实装 | `hal/pipeline.py` + `profiles/hygon_dcu.yaml` |
| Policy Pass（DeepSeek） | `compiler/passes/` Pass-Lite（非 IR 图优化） |
| IR 图 Pass（Full） | **Qwen3 阶段**（KV/GQA）；依赖 M2 Quant IR |
| Qwen3 GQA/KV/hgemm | `model/dense/` 分叉 + Compiler **Full Pass** + `export/` |
| 精度门禁 | `validate/` 新增 accuracy gate，挂 `QuantPipeline(stages=…)` 后置 |

---

## 4. 近期两周冲刺（8.24–9.10）

| 日期 | 产出 |
|------|------|
| 8.24–8.25 | 《技术选型分析调研报告》（框架=luban-sculpt；海光主路径=lmslim） |
| 8.25–8.30 | 《概要设计文档》：架构图、模块边界、`BackendPlan` 协议、异常/dry-run 约定 |
| 9.1–9.5 | 《详细设计》时序图 + 异常流；IR 初稿（Dense/MoE） |
| 9.6–9.10 | 测试框架选型报告；性能指标定义（延迟/吞吐/显存/精度阈值）；M1 评审 |

---

## 5. 风险与依赖

| 风险 | 影响 | 缓解 |
|------|------|------|
| Sourcefind **lmslim** wheel 与 DTK/torch 版本绑定 | 海光通路阻塞 | 8 月内锁定 DAS 版本；CI 用 dry-run + 一台 DCU 金机 |
| 「统一 IR」与现有 `BackendPlan` 双轨 | 工期膨胀 | IR 先服务 Pass/导出；短期 Backend 仍吃 `BackendPlan` |
| DeepSeek-MoE + AWQ 精度不达标 | M3 风险 | 默认 GPTQ/W8 路径；AWQ 作可选实验轨 |
| 72B 资源不足 | Qwen3 模测缩水 | 以 1.8B/7B 必验、72B 条件验收写入 M4 |
| Ignore 未贯通导致「策略假生效」 | 精度不可解释 | **9.11–10.5 列为底座 P0**，M2 必验 |

---

## 6. 原计划日期勘误

- 原表「Qwen3 里程碑 **11.3**」与开发窗 **11.3–11.30** 冲突 → 修订为 **M4 = 11.30**。  
- 原「基础算子库」在现架构中落地为 **lmslim 适配层 + HAL 编码/布局**，不另起一套无关 kernel 仓库（除非海光侧另有独立算子仓需求）。

---

## 8. DeepSeek 阶段任务勘误（原三行作废）

原排期下列表述与 **luban-sculpt + 海光 lmslim 整模 PTQ** 路径不符，**不再作为工作内容**：

| 原任务（作废） | 问题 | 替换为 |
|----------------|------|--------|
| 结构适配：专家**并行**量化策略、router FP16、shared expert | 「并行」易误解为训练/推理 EP；策略仅写在 `ArchQuantPolicy`，**未进 lmslim** | **MoE 策略执行**：模块名 + Ignore→exclude + expert-aware calib |
| HAE 迭代：算子融合提示 + **精度补偿**（对接 HAL） | 融合/补偿混在 HAE；HAL 仍为 stub；补偿应属 **calib V1** | **HAL 实装**（encoding/layout/kernel 选择）；补偿在校准行 |
| Compiler Pass：**IR 图插量化节点 + 图优化** | 海光主路径 **lmslim 内部整模 quant**，M2 前无 Quant IR；与 DeepSeek 窗口重复且空转 | **Policy Pass（Pass-Lite）**；**Full IR Pass 挪至 Qwen3**（KV/GQA） |

**DeepSeek 阶段正确主线**：`calib V1` → `MoE 策略执行` → `HAL 实装` → `Policy Pass` → `精度门禁` → `lmslim 通路测试` → **M3**。

---

## 7. 文档产出清单（按阶段）

| 阶段 | 文档 |
|------|------|
| 8.25 | 《技术选型分析调研报告》 |
| 8.30 | 《概要设计文档》（含架构图、Backend↔Compiler 协议） |
| 9.10 | 《详细设计文档》、测试框架选型报告、Quant IR 初稿 |
| 10.5 | M2 空转测试报告 |
| 11.2 | M3 DeepSeek 海光验证报告 |
| 11.30 | M4 Qwen3 通路验证报告 |
| 12.18–12.25 | 效果对比报告、性能测试报告、《海光量化调优指南》 |
| 12.30 | 终验打包说明 + 全量回归报告 |

---

*本表随 `luban_sculpt` 迭代更新；变更需同步里程碑验收标准。*
