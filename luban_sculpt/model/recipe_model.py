"""Recipe ``model`` 块：权重目录 / Hub ID + 期望目录布局。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from luban_sculpt.log import get_logger

logger = get_logger(__name__)

# layout 名 → 本地目录需满足的文件/通配说明（校验用）
LAYOUT_SPECS: dict[str, dict[str, Any]] = {
    "hf_pretrained": {
        "required_files": ("config.json",),
        "weight_globs": (
            "*.safetensors",
            "model*.safetensors",
            "pytorch_model.bin",
            "model*.bin",
        ),
        "notes": "Hugging Face 标准目录：config.json + tokenizer* + 权重分片",
    },
    "hub_id": {
        "required_files": (),
        "weight_globs": (),
        "notes": "Hub 模型 ID（如 org/name），不在本地校验文件",
    },
}


class ModelLayoutError(ValueError):
    """本地模型目录不符合声明的 layout。"""


def _is_local_model_path(raw: str) -> bool:
    if not raw or "://" in raw:
        return False
    p = Path(raw)
    return p.exists() and p.is_dir()


def _pick_model_ref(block: dict[str, Any]) -> str | None:
    for key in ("path", "dir", "id", "hub_id", "model_id"):
        val = block.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def validate_local_layout(model_dir: Path, layout: str) -> None:
    """按 layout 检查本地 HF 目录；缺权重仅 warn（便于 dry-run / 仅 config 探测）。"""
    spec = LAYOUT_SPECS.get(layout)
    if spec is None:
        raise ModelLayoutError(
            f"unknown model layout {layout!r}; known: {sorted(LAYOUT_SPECS)}"
        )
    missing = [name for name in spec["required_files"] if not (model_dir / name).is_file()]
    if missing:
        raise ModelLayoutError(
            f"model dir {model_dir} layout={layout!r} missing: {missing}"
        )
    weight_globs: tuple[str, ...] = spec.get("weight_globs") or ()
    if weight_globs:
        has_weights = any(model_dir.glob(g) for g in weight_globs)
        if not has_weights:
            logger.warning(
                "model dir %s layout=%s: no weight files matched %s "
                "(dry-run or download pending?)",
                model_dir,
                layout,
                weight_globs,
            )


def normalize_recipe_model(recipe: dict[str, Any], *, validate_layout: bool = True) -> dict[str, Any]:
    """将 ``model: { path, arch, layout }`` 展开为 ``model_id`` / ``model_arch``。"""
    out = dict(recipe)
    block = out.get("model")
    model_id: str | None = out.get("model_id")

    if isinstance(block, dict):
        ref = _pick_model_ref(block)
        if ref:
            model_id = ref
            out["model_id"] = ref
        arch = block.get("arch") or block.get("model_arch")
        if arch is not None:
            out["model_arch"] = arch
        layout = str(block.get("layout") or "hf_pretrained").strip()
        if model_id and _is_local_model_path(model_id):
            if validate_layout and layout != "hub_id":
                validate_local_layout(Path(model_id).resolve(), layout)
        elif model_id and layout == "hf_pretrained" and not _is_local_model_path(model_id):
            # Hub / 相对路径尚未 materialize：视为 hub_id
            pass
    if not out.get("model_id"):
        raise ValueError(
            "recipe requires model.path (or model_id); "
            "see recipes/*.yaml model: block"
        )
    return out


def apply_model_path_override(recipe: dict[str, Any], path: str) -> dict[str, Any]:
    """CLI --model-dir / --model-id：写入 model.path 并同步 model_id。"""
    out = dict(recipe)
    p = str(path).strip()
    block = dict(out["model"]) if isinstance(out.get("model"), dict) else {}
    block["path"] = p
    block.setdefault("layout", "hf_pretrained" if _is_local_model_path(p) else "hub_id")
    out["model"] = block
    out["model_id"] = p
    return out
