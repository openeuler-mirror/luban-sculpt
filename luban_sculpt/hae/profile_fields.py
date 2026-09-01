"""Profile YAML field helpers (single source: capability + soc_key).

Data lives under ``luban_sculpt/profiles/*.yaml``; this module only reads it.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml


def profiles_dir() -> Path:
    from importlib.resources import files

    return Path(str(files("luban_sculpt").joinpath("profiles")))


@functools.lru_cache(maxsize=32)
def load_profile_doc(profile_id: str) -> dict[str, Any] | None:
    path = profiles_dir() / f"{profile_id}.yaml"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    return doc if isinstance(doc, dict) else None


def profile_soc_key(profile: dict[str, Any]) -> str | None:
    raw = profile.get("soc_key")
    if raw:
        return str(raw)
    chip = profile.get("chip") or {}
    if chip.get("soc_key"):
        return str(chip["soc_key"])
    if chip.get("soc"):
        return str(chip["soc"])
    return None


def profile_fp8_native(profile: dict[str, Any]) -> bool:
    cap = profile.get("capability") or {}
    if "fp8_native" in cap:
        return bool(cap["fp8_native"])
    chip = profile.get("chip") or {}
    if "fp8_native" in chip:
        return bool(chip["fp8_native"])
    gpu = profile.get("gpu") or {}
    if "fp8_native" in gpu:
        return bool(gpu["fp8_native"])
    return False


def profile_match_keys(profile: dict[str, Any]) -> list[str]:
    keys = profile.get("match_keys")
    if keys:
        return [str(k).lower() for k in keys]
    sk = profile_soc_key(profile)
    return [sk.lower()] if sk else []


def infer_expected_deploy(scheme_cfg: dict[str, Any]) -> str | None:
    infer = scheme_cfg.get("infer") or {}
    return infer.get("deploy_target") or infer.get("runtime")
