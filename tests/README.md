# Test layout mirrors ``luban_sculpt/`` modules.

| 目录 | 对应模块 |
|------|----------|
| `tests/calib/` | `luban_sculpt.calib` |
| `tests/pipeline/` | `luban_sculpt.pipeline` |
| `tests/compiler/` | `luban_sculpt.compiler` |
| `tests/model/` | `luban_sculpt.model` |
| `tests/hae/` | `luban_sculpt.hae` |
| `tests/validate/` | `luban_sculpt.validate` |
| `tests/backends/` | `luban_sculpt.backends.*` |
| `tests/log/` | `luban_sculpt.log` |
| `tests/cli/` | `luban_sculpt.cli` |
| `tests/e2e/` | Dry-run 完整链路（probe→compress→validate→report） |
| `tests/paths.py` | 仓库根 / recipes / profiles 路径 |

```bash
# 全部
pytest

# 按模块
pytest tests/calib tests/pipeline -q

# dry-run 端到端
pytest tests/e2e -q
```
