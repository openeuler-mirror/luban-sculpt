"""Recipe ``extends: quant`` → 合并 ``templates/*.yaml``。"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from luban_sculpt.log import get_logger

logger = get_logger(__name__)


def _templates_dir() -> Path:
    return Path(str(files("luban_sculpt").joinpath("templates")))


def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def _merge_pipeline(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    if "stages" in patch:
        base_stages = list(out.get("stages") or [])
        over_stages = patch["stages"]
        if not isinstance(over_stages, list):
            raise TypeError("pipeline.stages must be a list")
        merged: list[Any] = []
        for i, ost in enumerate(over_stages):
            if not isinstance(ost, dict):
                merged.append(ost)
                continue
            if i < len(base_stages) and isinstance(base_stages[i], dict):
                merged.append(_deep_merge_dict(base_stages[i], ost))
            else:
                merged.append(dict(ost))
        if len(base_stages) > len(over_stages):
            merged.extend(base_stages[len(over_stages) :])
        out["stages"] = merged
    for key, val in patch.items():
        if key != "stages":
            out[key] = val
    return out


def merge_recipe_documents(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并 recipe；``pipeline.stages[i]`` 按 index 与模板 stage 合并。"""
    out = dict(base)
    for key, val in override.items():
        if key == "pipeline" and isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_pipeline(out[key], val)
        elif isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def _load_template_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"template must be a mapping: {path}")
    # 模板自身不可再 extends，避免循环
    doc.pop("extends", None)
    return doc


def resolve_template_name(name: str) -> Path:
    """``quant`` / ``quant.yaml`` / ``templates/quant.yaml`` → 包内路径。"""
    raw = name.strip().replace("\\", "/")
    if raw.startswith("templates/"):
        raw = raw[len("templates/") :]
    if not raw.endswith(".yaml"):
        raw = f"{raw}.yaml"
    path = _templates_dir() / raw
    if not path.is_file():
        raise FileNotFoundError(f"recipe template not found: {name} ({path})")
    return path


def load_recipe_template(name: str) -> dict[str, Any]:
    path = resolve_template_name(name)
    logger.debug("load recipe template %s from %s", name, path)
    return _load_template_file(path)


def resolve_recipe_extends(
    doc: dict[str, Any],
    *,
    recipe_path: Path | None = None,
) -> dict[str, Any]:
    """若含 ``extends``，与模板合并后返回（``extends`` 键已移除）。"""
    extends = doc.pop("extends", None)
    if not extends:
        return doc
    base = load_recipe_template(str(extends))
    merged = merge_recipe_documents(base, doc)
    logger.info(
        "recipe extends=%s path=%s",
        extends,
        recipe_path,
    )
    return merged
