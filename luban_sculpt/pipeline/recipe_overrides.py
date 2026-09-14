"""CLI / 环境变量覆盖通用 recipe 模板。"""

from __future__ import annotations

from typing import Any

from luban_sculpt.model.recipe_model import apply_model_path_override, normalize_recipe_model
from luban_sculpt.pipeline.recipe_presets import (
    CALIB_PRESETS,
    PIPELINE_PRESETS,
    PRECISION_OVERLAYS,
)


def _deep_merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def _merge_compress_overlay(stage: dict[str, Any], patch: dict[str, Any]) -> None:
    if not patch:
        return
    existing = stage.get("compress")
    if isinstance(existing, dict):
        stage["compress"] = _deep_merge_dict(existing, patch)
    else:
        stage["compress"] = dict(patch)


def _apply_precision_overlay(
    out: dict[str, Any],
    stages: list[dict[str, Any]],
    precision_key: str,
    stage_index: int = 0,
) -> None:
    overlay = PRECISION_OVERLAYS.get(precision_key)
    if not overlay:
        return
    if "calib" in overlay:
        out["calib"] = _deep_merge_dict(dict(out.get("calib") or {}), overlay["calib"])
    if stage_index < len(stages):
        _merge_compress_overlay(stages[stage_index], overlay.get("compress") or {})


def apply_recipe_cli_overrides(
    recipe: dict[str, Any],
    *,
    model_id: str | None = None,
    model_dir: str | None = None,
    validate_model_layout: bool = True,
    precision: str | None = None,
    abstract_scheme: str | None = None,
    backend: str | None = None,
    ignore: list[str] | None = None,
    calib_preset: str | None = None,
    calib: dict[str, Any] | None = None,
    pipeline_preset: str | None = None,
    stage_index: int = 0,
) -> dict[str, Any]:
    """合并 CLI 参数到 recipe（模型、精度、ignore、校准、多阶段 preset）。"""
    out = dict(recipe)
    path_override = model_dir or model_id
    if path_override:
        out = apply_model_path_override(out, path_override)

    if pipeline_preset:
        preset = PIPELINE_PRESETS.get(pipeline_preset)
        if preset is None:
            known = ", ".join(sorted(PIPELINE_PRESETS))
            raise ValueError(f"unknown pipeline preset {pipeline_preset!r}; choose: {known}")
        pipeline = dict(out.get("pipeline") or {})
        pipeline["stages"] = [dict(s) for s in preset["stages"]]
        out["pipeline"] = pipeline
        if "calib" in preset:
            out["calib"] = _deep_merge_dict(dict(out.get("calib") or {}), preset["calib"])
        if calib_preset:
            cp = CALIB_PRESETS.get(calib_preset)
            if cp:
                out["calib"] = _deep_merge_dict(out["calib"], cp["calib"])
        if calib:
            out["calib"] = _deep_merge_dict(dict(out.get("calib") or {}), calib)
        return normalize_recipe_model(out, validate_layout=validate_model_layout)

    has_stage = any((precision, abstract_scheme, backend, ignore, calib_preset))
    has_top = calib is not None
    if not has_stage and not has_top and not path_override:
        return normalize_recipe_model(out, validate_layout=validate_model_layout)

    if calib_preset:
        preset = CALIB_PRESETS.get(calib_preset)
        if preset is None:
            known = ", ".join(sorted(CALIB_PRESETS))
            raise ValueError(f"unknown calib preset {calib_preset!r}; choose: {known}")
        out["calib"] = _deep_merge_dict(dict(out.get("calib") or {}), preset["calib"])
    if calib:
        out["calib"] = _deep_merge_dict(dict(out.get("calib") or {}), calib)

    pipeline = dict(out.get("pipeline") or {})
    stages_raw = pipeline.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        return normalize_recipe_model(out, validate_layout=validate_model_layout)

    stages = [dict(s) if isinstance(s, dict) else s for s in stages_raw]
    if stage_index >= len(stages) or not isinstance(stages[stage_index], dict):
        return normalize_recipe_model(out, validate_layout=validate_model_layout)

    st = stages[stage_index]
    if backend:
        st["backend"] = backend
    if abstract_scheme:
        st["abstract_scheme"] = abstract_scheme
        st.pop("precision", None)
    elif precision:
        st["precision"] = precision
        st.pop("abstract_scheme", None)
    if ignore is not None:
        st["ignore"] = list(ignore)

    if calib_preset:
        preset = CALIB_PRESETS[calib_preset]
        _merge_compress_overlay(st, preset.get("compress") or {})

    effective_precision = str(st.get("precision") or "")
    if precision:
        effective_precision = precision
    _apply_precision_overlay(out, stages, effective_precision, stage_index)

    stages[stage_index] = st
    pipeline["stages"] = stages
    out["pipeline"] = pipeline
    return normalize_recipe_model(out, validate_layout=validate_model_layout)


apply_compress_cli_overrides = apply_recipe_cli_overrides
