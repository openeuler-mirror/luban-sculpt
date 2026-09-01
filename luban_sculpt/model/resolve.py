"""Resolve model architecture from model_id / HF config into ModelArchSnapshot (MoE before dense)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from luban_sculpt.model import dense as dense_mod
from luban_sculpt.model import moe as moe_mod
from luban_sculpt.model.presets import ARCH_POLICIES, get_arch_policy
from luban_sculpt.model.types import MOE_ARCHES, ModelArch, ModelArchSnapshot

logger = logging.getLogger(__name__)

# MoE first so qwen2_moe / DeepSeek-V3 win over dense qwen/deepseek
_HF_MODEL_TYPE_MAP: dict[str, ModelArch] = {
    **dense_mod.HF_MODEL_TYPE_MAP,
    **moe_mod.HF_MODEL_TYPE_MAP,
}
_ARCH_PATTERNS = list(moe_mod.ARCH_PATTERNS) + list(dense_mod.ARCH_PATTERNS)
_ID_PATTERNS = list(moe_mod.ID_PATTERNS) + list(dense_mod.ID_PATTERNS)

_MULTIMODAL_RECIPE_KEYS = frozenset({"multimodal", "vl", "vision", "mm"})

_ARCH_ALIASES = {
    "llama2": "llama",
    "llama3": "llama",
    "llama3_1": "llama",
    "qwen2": "qwen",
    "qwen2_5": "qwen",
    "qwen3": "qwen",
    "qwen_moe": "qwen_moe",
    "dense": "dense_generic",
    "moe": "moe_generic",
    "baichuan": "dense_generic",
    "yi": "dense_generic",
    "phi": "dense_generic",
    "phi3": "dense_generic",
    "internlm": "dense_generic",
    "internlm2": "dense_generic",
}

_MULTIMODAL_IGNORE: tuple[str, ...] = (
    "visual.*",
    "vision_tower.*",
    "multi_modal_projector.*",
    "re:.*vision.*",
)

_MULTIMODAL_QUANT_HINTS: dict[str, Any] = {
    "quantize_vision": False,
    "text_only_ptq": True,
}

_VL_HF_TYPE_RE = re.compile(r"_vl$|_vl_|vl2", re.I)
_VL_ARCH_RE = re.compile(r"Qwen2VL|Qwen2_5_VL|Llava|InternVL", re.I)
_VL_ID_RE = re.compile(r"qwen.*vl|vl.*qwen|llava|internvl", re.I)


def _parse_arch(raw: str | ModelArch | None) -> ModelArch | None:
    if raw is None:
        return None
    if isinstance(raw, ModelArch):
        return raw
    key = str(raw).strip().lower().replace("-", "_")
    if key in _MULTIMODAL_RECIPE_KEYS:
        return None
    key = _ARCH_ALIASES.get(key, key)
    try:
        return ModelArch(key)
    except ValueError:
        logger.warning("Unknown model_arch %r; falling back to auto", raw)
        return None


def _recipe_requests_multimodal(explicit: Any, model_block: dict[str, Any]) -> bool:
    if model_block.get("is_multimodal"):
        return True
    if explicit is None:
        return False
    key = str(explicit).strip().lower().replace("-", "_")
    return key in _MULTIMODAL_RECIPE_KEYS


def _infer_multimodal(
    *,
    model_id: str,
    hf_type: str | None,
    architectures: list[str],
    recipe_flag: bool,
) -> bool:
    if recipe_flag:
        return True
    if hf_type and _VL_HF_TYPE_RE.search(hf_type):
        return True
    if any(_VL_ARCH_RE.search(name) for name in architectures):
        return True
    blob = model_id.replace("\\", "/")
    return bool(_VL_ID_RE.search(blob))


def _read_hf_config(model_id: str) -> dict[str, Any] | None:
    path = Path(model_id)
    cfg_path = path / "config.json" if path.is_dir() else None
    if cfg_path is None or not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed reading %s: %s", cfg_path, exc)
        return None


def _arch_from_hf_config(cfg: dict[str, Any]) -> tuple[ModelArch | None, str | None, list[str]]:
    hf_type = cfg.get("model_type")
    arches = [str(a) for a in (cfg.get("architectures") or [])]
    if isinstance(hf_type, str):
        mapped = _HF_MODEL_TYPE_MAP.get(hf_type.lower())
        if mapped:
            return mapped, hf_type, arches
    for arch_name in arches:
        for pat, arch in _ARCH_PATTERNS:
            if pat.search(arch_name):
                return arch, hf_type if isinstance(hf_type, str) else None, arches
    return None, hf_type if isinstance(hf_type, str) else None, arches


def _arch_from_model_id(model_id: str) -> ModelArch | None:
    name = model_id.replace("\\", "/")
    base = name.rsplit("/", 1)[-1]
    blob = f"{name} {base}"
    for pat, arch in _ID_PATTERNS:
        if pat.search(blob):
            return arch
    return None


def resolve_model_arch(
    model_id: str,
    *,
    recipe: dict[str, Any] | None = None,
    hf_config: dict[str, Any] | None = None,
) -> ModelArchSnapshot:
    """Resolve ModelArchSnapshot: recipe override → HF config → model_id heuristic."""
    recipe = recipe or {}
    model_block = recipe.get("model") if isinstance(recipe.get("model"), dict) else {}
    explicit = (
        recipe.get("model_arch")
        or model_block.get("arch")
        or model_block.get("model_arch")
    )
    mm_from_recipe = _recipe_requests_multimodal(explicit, model_block)
    arch = _parse_arch(explicit)
    if arch is not None:
        source = "recipe"
    else:
        source = "unknown"
    hf_type: str | None = None
    arches: list[str] = []

    cfg = hf_config if hf_config is not None else _read_hf_config(model_id)
    if cfg:
        cfg_arch, hf_type, arches = _arch_from_hf_config(cfg)
        if arch is None and cfg_arch is not None:
            arch = cfg_arch
            source = "hf_config"

    if arch is None:
        arch = _arch_from_model_id(model_id) or ModelArch.UNKNOWN
        source = "heuristic" if arch != ModelArch.UNKNOWN else "unknown"

    if mm_from_recipe and arch in (ModelArch.UNKNOWN, ModelArch.DENSE_GENERIC):
        arch = ModelArch.QWEN
        if source == "unknown":
            source = "recipe"

    policy = get_arch_policy(arch)
    is_moe = bool(model_block.get("is_moe", arch in MOE_ARCHES))
    is_mm = _infer_multimodal(
        model_id=model_id,
        hf_type=hf_type,
        architectures=arches,
        recipe_flag=mm_from_recipe,
    )
    if arch in MOE_ARCHES:
        is_moe = True

    quant_hints = dict(policy.quant_hints)
    if is_mm:
        quant_hints = {**quant_hints, **_MULTIMODAL_QUANT_HINTS}

    return ModelArchSnapshot(
        arch=arch,
        is_moe=is_moe,
        is_multimodal=is_mm,
        hf_model_type=hf_type,
        architectures=arches,
        source=source,
        quant_hints=quant_hints,
    )


def merge_ignore_list(
    recipe_ignore: list[str] | None,
    arch: ModelArch,
    *,
    is_multimodal: bool = False,
) -> list[str]:
    """Recipe ignore first, then arch policy defaults (dedupe, preserve order)."""
    policy = get_arch_policy(arch)
    seen: set[str] = set()
    out: list[str] = []
    extras: tuple[str, ...] = ()
    if is_multimodal:
        extras = _MULTIMODAL_IGNORE
    for item in list(recipe_ignore or []) + list(policy.default_ignore) + list(extras):
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def apply_arch_to_backend_options(
    backend: str,
    options: dict[str, Any],
    model: ModelArchSnapshot,
) -> dict[str, Any]:
    """Inject arch-derived defaults into backend_options (non-destructive)."""
    opts = dict(options or {})
    policy = get_arch_policy(model.arch)
    opts.setdefault("model_arch", model.arch.value)
    opts.setdefault("is_moe", model.is_moe)
    opts.setdefault("is_multimodal", model.is_multimodal)

    if backend == "msmodelslim" and not opts.get("model_type"):
        hint = policy.quant_hints.get("msmodelslim_model_type")
        if hint:
            opts["model_type"] = str(hint)

    if model.quant_hints:
        opts.setdefault("arch_quant_hints", model.quant_hints)

    return opts


def list_arches() -> list[dict[str, Any]]:
    """CLI / docs helper: arch → policy summary (MoE section then dense)."""
    rows: list[dict[str, Any]] = []
    moe_items = [(a, p) for a, p in ARCH_POLICIES.items() if a in MOE_ARCHES]
    dense_items = [(a, p) for a, p in ARCH_POLICIES.items() if a not in MOE_ARCHES]
    for arch, policy in moe_items + dense_items:
        rows.append(
            {
                "arch": arch.value,
                "group": "moe" if arch in MOE_ARCHES else "dense",
                "is_moe": arch in MOE_ARCHES,
                "default_ignore": list(policy.default_ignore),
                "quant_hints": dict(policy.quant_hints),
                "notes": policy.notes,
            }
        )
    return rows
