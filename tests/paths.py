"""Shared paths for luban_sculpt tests (repo root / package / recipes)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "luban_sculpt"
RECIPES = PKG / "recipes"
TEMPLATES = PKG / "templates"
LLAMA3_EXAMPLE = RECIPES / "llama3.yaml"
QWEN25_EXAMPLE = RECIPES / "qwen2_5_7b.yaml"
LLAMA3_FP8_THEN_GPTQ = RECIPES / "llama3_fp8_then_gptq.yaml"
PROFILES = PKG / "profiles"
