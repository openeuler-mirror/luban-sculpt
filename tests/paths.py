"""Shared paths for luban_sculpt tests (repo root / package / recipes)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "luban_sculpt"
RECIPES = PKG / "recipes"
PROFILES = PKG / "profiles"
