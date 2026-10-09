"""补充 vLLM 融合模块的量化忽略名称。"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from luban_sculpt.log import get_logger

logger = get_logger(__name__)

# 架构名称 → 融合模块后缀 → 原始模块后缀
#
# 仅登记已确认的融合关系。
# 只有所有原始模块都被明确忽略，才补充融合模块名称。
FUSED_MODULE_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "KimiLinearForCausalLM": {
        "shared_experts.gate_up_proj": (
            "shared_experts.gate_proj",
            "shared_experts.up_proj",
        ),
    },
}


def _is_exact_module_name(name: str) -> bool:
    """只使用明确的模块名称，不展开正则或通配符规则。"""
    return (
        bool(name)
        and not name.startswith("re:")
        and not any(char in name for char in "*?[]")
    )


def _write_config_atomic(config_path: Path, config: dict) -> None:
    """使用同目录临时文件替换配置，并保留原文件权限。"""
    permissions = stat.S_IMODE(config_path.stat().st_mode)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        os.chmod(temp_path, permissions)
        os.replace(temp_path, config_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def patch_vllm_fused_ignore(model_dir: str | Path) -> list[str]:
    """为 compressed-tensors 配置补充 vLLM 融合模块忽略项。

    只修改 config.json，不读取或修改权重文件。

    未登记的架构、其他量化格式、没有配置的 dry-run 目录
    均直接跳过。重复执行不会重复添加名称。

    返回本次新增的融合模块名称。
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return []

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"模型配置必须是 JSON 对象: {config_path}")

    architectures = config.get("architectures", [])
    if not isinstance(architectures, list):
        return []

    rules: dict[str, tuple[str, ...]] = {}
    for arch in architectures:
        if isinstance(arch, str):
            rules.update(FUSED_MODULE_RULES.get(arch, {}))

    if not rules:
        return []

    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return []

    if quant_config.get("quant_method") != "compressed-tensors":
        return []

    ignore = quant_config.get("ignore")
    if ignore is None:
        return []

    if not isinstance(ignore, list) or not all(
        isinstance(name, str) for name in ignore
    ):
        raise ValueError(
            f"quantization_config.ignore 必须是字符串列表: {config_path}"
        )

    # 使用原始快照作为判断依据，避免新增别名影响其他规则。
    exact_ignored = {
        name for name in ignore if _is_exact_module_name(name)
    }
    existing = set(ignore)
    added: list[str] = []

    for fused_suffix, original_suffixes in rules.items():
        first_suffix = "." + original_suffixes[0]

        for name in sorted(exact_ignored):
            if not name.endswith(first_suffix):
                continue

            # 例如 model.layers.1.block_sparse_moe
            prefix = name[:-len(first_suffix)]
            if not prefix:
                continue

            original_names = [
                f"{prefix}.{suffix}" for suffix in original_suffixes
            ]

            if not all(
                original in exact_ignored for original in original_names
            ):
                continue

            fused_name = f"{prefix}.{fused_suffix}"
            if fused_name in existing:
                continue

            ignore.append(fused_name)
            existing.add(fused_name)
            added.append(fused_name)

    if added:
        _write_config_atomic(config_path, config)
        logger.info(
            "Added %d vLLM fused ignore aliases in %s: %s",
            len(added),
            config_path,
            added[:5],
        )

    return added