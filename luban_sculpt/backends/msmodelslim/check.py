"""Ascend SoC quant-type policy for msModelSlim (loaded from profiles/*.yaml)."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import yaml

from luban_sculpt.hae.profile_fields import (
    load_profile_doc,
    profile_fp8_native,
    profile_soc_key,
    profiles_dir,
)


@dataclass(frozen=True)
class AscendChipSpec:
    profile_name: str
    soc_key: str
    soc: str
    fp8_native: bool
    allowed_quant_types: tuple[str, ...]
    default_quant_type: str
    default_abstract_scheme: str
    peak_tflops_fp16: float | None = None
    vllm_quantization: str = "ascend"
    notes: str = ""


def _display_name(doc: dict) -> str:
    name = doc.get("display_name")
    if name:
        return str(name)
    sk = profile_soc_key(doc) or doc.get("id", "Ascend")
    return f"Ascend {str(sk).upper()}"


def _spec_from_doc(doc: dict) -> AscendChipSpec | None:
    soc_key = profile_soc_key(doc)
    if not soc_key:
        return None
    ms = doc.get("msmodelslim") or {}
    allowed = ms.get("allowed_quant_types")
    if not allowed:
        return None
    cap = doc.get("capability") or {}
    peak = cap.get("peak_tflops_fp16")
    return AscendChipSpec(
        profile_name=str(doc.get("id") or soc_key),
        soc_key=str(soc_key).lower(),
        soc=_display_name(doc),
        fp8_native=profile_fp8_native(doc),
        allowed_quant_types=tuple(str(x).lower() for x in allowed),
        default_quant_type=str(ms.get("default_quant_type", "w8a8")).lower(),
        default_abstract_scheme=str(
            ms.get("default_abstract_scheme", "ascend_w8a8")
        ),
        peak_tflops_fp16=float(peak) if peak is not None else None,
        notes=str(ms.get("notes") or ""),
    )


@functools.lru_cache(maxsize=1)
def _ascend_chips_index() -> dict[str, AscendChipSpec]:
    chips: dict[str, AscendChipSpec] = {}
    root = profiles_dir()
    for path in sorted(root.glob("ascend_*.yaml")):
        with path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            continue
        spec = _spec_from_doc(doc)
        if spec:
            chips[spec.soc_key] = spec
    return chips


def clear_ascend_chips_cache() -> None:
    _ascend_chips_index.cache_clear()


class _AscendChipsProxy(dict):
    """Lazy dict so tests can patch after import without stale hardcoded table."""

    def __getitem__(self, key: str) -> AscendChipSpec:
        return _ascend_chips_index()[key]

    def __iter__(self):
        return iter(_ascend_chips_index())

    def __len__(self) -> int:
        return len(_ascend_chips_index())

    def values(self):
        return _ascend_chips_index().values()

    def keys(self):
        return _ascend_chips_index().keys()

    def items(self):
        return _ascend_chips_index().items()

    def get(self, key: str, default=None):
        return _ascend_chips_index().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in _ascend_chips_index()


ASCEND_CHIPS: _AscendChipsProxy = _AscendChipsProxy()


def normalize_soc_key(raw: str) -> str | None:
    s = raw.strip().lower().replace("ascend", "").replace("_", "").replace("-", "")
    if not s:
        return None
    index = _ascend_chips_index()
    for key in index:
        if key in s or s.endswith(key):
            return key
    aliases = {
        "910b1": "910b",
        "910b2": "910b",
        "910b3": "910b",
        "910b4": "910b",
    }
    aliased = aliases.get(s)
    if aliased and aliased in index:
        return aliased
    return None


def get_chip(spec_key: str) -> AscendChipSpec:
    key = normalize_soc_key(spec_key) or spec_key.lower()
    index = _ascend_chips_index()
    if key not in index:
        doc = load_profile_doc(f"ascend_{key}")
        if doc:
            spec = _spec_from_doc(doc)
            if spec:
                return spec
        raise KeyError(f"Unknown Ascend SoC {spec_key!r}; known: {list(index)}")
    return index[key]


def assert_quant_type_allowed(chip: AscendChipSpec, quant_type: str) -> None:
    qt = quant_type.lower()
    if qt == "fp8" and not chip.fp8_native:
        raise ValueError(
            f"{chip.soc} 不支持 FP8 原生量化；allowed={chip.allowed_quant_types}. "
            f"{chip.notes}".strip()
        )
    if qt not in chip.allowed_quant_types and qt != "fp8":
        raise ValueError(
            f"quant_type {quant_type!r} not allowed on {chip.soc}; "
            f"allowed={chip.allowed_quant_types}"
        )


def scheme_implies_fp8(abstract_scheme: str) -> bool:
    return abstract_scheme in ("ascend_fp8", "fp8_dynamic", "fp8_native")
