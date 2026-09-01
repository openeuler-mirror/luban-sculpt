"""Calibration dataset helpers (shared by all backends via CalibRunner)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator

from luban_sculpt.log import get_logger

logger = get_logger(__name__)

# 包内默认 open-perfectblend（datasets save_to_disk）
_PKG_CALIB_ROOT = Path(__file__).resolve().parent
_DEFAULT_PERFECTBLEND = _PKG_CALIB_ROOT / "open-perfectblend"

ROLE_MAP = {"human": "user", "gpt": "assistant"}

# source 别名 → 目录
_SOURCE_ALIASES: dict[str, Path] = {
    "open-perfectblend": _DEFAULT_PERFECTBLEND,
    "perfectblend": _DEFAULT_PERFECTBLEND,
    "open_perfectblend": _DEFAULT_PERFECTBLEND,
}


def default_perfectblend_dir() -> Path:
    return _DEFAULT_PERFECTBLEND


def resolve_calib_path(
    source: str,
    *,
    path: str | Path | None = None,
) -> Path | None:
    """解析本地校准目录：显式 path > source 别名 > source 若为已存在目录。"""
    if path:
        p = Path(path).expanduser().resolve()
        return p if p.exists() else p
    alias = _SOURCE_ALIASES.get(source)
    if alias is not None:
        return alias
    if source and source not in ("stub",):
        p = Path(source).expanduser()
        if p.exists() and (p.is_dir() or p.is_file()):
            return p.resolve()
    return None


def conversations_to_text(conversations: Any) -> str:
    """perfectblend ``conversations`` → 纯文本（无 tokenizer 时的简易格式）。"""
    if not conversations:
        return ""
    lines: list[str] = []
    for m in conversations:
        if not isinstance(m, dict):
            continue
        role = ROLE_MAP.get(m.get("from", "user"), m.get("from", "user"))
        content = m.get("value", "") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def row_to_text(row: dict[str, Any], text_column: str = "text") -> str:
    if text_column in row and row[text_column]:
        return str(row[text_column])
    if "conversations" in row:
        return conversations_to_text(row["conversations"])
    if "text" in row:
        return str(row["text"] or "")
    return str(row)


def truncate_text(text: str, max_chars: int | None) -> str:
    """按字符粗截断，避免超长校准样本把 GPTQ Hessian / KV 打爆。"""
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _sample_indices(n_total: int, max_samples: int, *, shuffle: bool, seed: int) -> list[int]:
    """只打乱下标，避免对整表 ``Dataset.shuffle()``（大库会拖垮内存）。"""
    n = min(max_samples, n_total)
    if n <= 0:
        return []
    if not shuffle:
        return list(range(n))
    rng = random.Random(seed)
    # 大库：随机采样下标；小库：全量 shuffle 再截断
    if n_total > max(n * 8, 4096):
        return rng.sample(range(n_total), n)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    return idxs[:n]


def _iter_from_disk(
    dataset_dir: Path,
    *,
    max_samples: int,
    split: str,
    text_column: str,
    shuffle: bool,
    seed: int,
    max_chars: int | None,
) -> Iterator[dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_dir))
    if hasattr(ds, "keys") and split in list(ds.keys()):
        ds = ds[split]
    elif hasattr(ds, "keys") and "train" in list(ds.keys()):
        ds = ds["train"]

    n_total = len(ds)
    idxs = _sample_indices(n_total, max_samples, shuffle=shuffle, seed=seed)
    logger.info(
        "calib disk sample n_total=%d take=%d shuffle=%s max_chars=%s path=%s",
        n_total,
        len(idxs),
        shuffle,
        max_chars,
        dataset_dir,
    )
    # 按索引取行，不做全表 shuffle / select 物化
    for i in idxs:
        row = ds[int(i)]
        text = truncate_text(row_to_text(row, text_column), max_chars)
        yield {"text": text}


def iter_text_samples(
    source: str = "stub",
    *,
    max_samples: int = 512,
    split: str = "train",
    text_column: str = "text",
    path: str | Path | None = None,
    shuffle: bool = True,
    seed: int = 42,
    max_chars: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield ``{"text": ...}``。

    - ``source=stub``：占位文本
    - ``source=open-perfectblend``（或别名）：读包内 ``calib/open-perfectblend``
    - ``path=...``：任意 ``save_to_disk`` / 含 arrow 的目录（覆盖 source 目录）
    - 其它：HF hub id（``load_dataset``），或本地已存在路径

    ``max_chars``：文本字符上限（粗截断）。GPTQ 建议配合 ``max_seq_length``。
    """
    local = resolve_calib_path(source, path=path)
    if local is not None:
        if not local.exists():
            logger.warning("calib path %s missing; fallback stub", local)
            yield from _stub(max_samples)
            return
        try:
            logger.info(
                "calib load_from_disk path=%s max_samples=%d",
                local,
                max_samples,
            )
            yield from _iter_from_disk(
                local,
                max_samples=max_samples,
                split=split,
                text_column=text_column,
                shuffle=shuffle,
                seed=seed,
                max_chars=max_chars,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "calib load_from_disk %s failed (%s); fallback",
                local,
                exc,
            )

    if source == "stub" or not source:
        yield from _stub(max_samples)
        return

    try:
        from datasets import load_dataset

        ds = load_dataset(source, split=split)
        n_total = len(ds)
        idxs = _sample_indices(n_total, max_samples, shuffle=shuffle, seed=seed)
        for i in idxs:
            row = ds[int(i)]
            yield {"text": truncate_text(row_to_text(row, text_column), max_chars)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("calib dataset %r load failed (%s); fallback stub", source, exc)
        yield from _stub(max_samples)


def _stub(max_samples: int) -> Iterator[dict[str, Any]]:
    for i in range(min(8, max_samples)):
        yield {"text": f"calib stub sample {i}"}
