"""Scheme loader branch: honor manifest producer / profile_id (stub)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_quant_config_with_producer(model_dir: Path) -> dict[str, Any] | None:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    producer = manifest.get("producer", {})
    if producer.get("name") != "luban-sculpt":
        return None
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return {"manifest_only": True, **manifest}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("_luban", {})
    config["_luban"]["profile_id"] = manifest.get("profile_id")
    config["_luban"]["export_format"] = manifest.get("export_format")
    return config
