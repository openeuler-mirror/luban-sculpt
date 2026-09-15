"""Recipe model block normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luban_sculpt.model.recipe_model import (
    ModelLayoutError,
    normalize_recipe_model,
    validate_local_layout,
)


def test_normalize_model_block_to_model_id(tmp_path: Path) -> None:
    model_dir = tmp_path / "llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")

    out = normalize_recipe_model(
        {
            "model": {"path": str(model_dir), "arch": "llama", "layout": "hf_pretrained"},
            "pipeline": {"stages": [{"backend": "auto", "precision": "fp8_dynamic"}]},
        }
    )
    assert out["model_id"] == str(model_dir)
    assert out["model_arch"] == "llama"


def test_layout_requires_config_json(tmp_path: Path) -> None:
    model_dir = tmp_path / "bad"
    model_dir.mkdir()
    with pytest.raises(ModelLayoutError, match="config.json"):
        validate_local_layout(model_dir, "hf_pretrained")
