# templates/

通用量化 **模板**，不绑定具体模型。`recipes/*.yaml` 使用：

```yaml
extends: quant
model:
  path: ./your-model
  arch: llama
```

| 文件 | 说明 |
|------|------|
| `quant.yaml` | 默认单阶段 `backend: auto`、`precision: fp8_dynamic`、QuantizationPatch、stub calib |

精度 / 校准 / 多阶段由 recipe 注释中的 CLI 或 `recipes/` 内字段指定。
