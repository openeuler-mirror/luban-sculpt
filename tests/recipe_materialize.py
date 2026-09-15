"""Write merged recipe docs to temp paths (extends + CLI overrides)."""

from __future__ import annotations

from pathlib import Path

import yaml

from luban_sculpt.compiler.recipe_compiler import load_recipe_yaml
from luban_sculpt.pipeline.recipe_overrides import apply_recipe_cli_overrides


def materialize_recipe(
    base_path: Path,
    dest: Path,
    *,
    precision: str | None = None,
) -> Path:
    doc = load_recipe_yaml(base_path, validate_model_layout=False)
    if precision:
        doc = apply_recipe_cli_overrides(doc, precision=precision)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dest
