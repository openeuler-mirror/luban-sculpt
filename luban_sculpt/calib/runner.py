"""Shared calibration runner for all quant backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from luban_sculpt.calib.datasets import (
    default_perfectblend_dir,
    iter_text_samples,
    resolve_calib_path,
)
from luban_sculpt.contracts import BackendPlan
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


@dataclass
class CalibData:
    """Materialized calib samples + report for a backend to consume."""

    samples: list[dict[str, Any]]
    texts: list[str]
    report: dict[str, Any] = field(default_factory=dict)
    resolved_path: str | None = None

    @property
    def num_samples(self) -> int:
        return len(self.samples)


@dataclass
class CalibRunner:
    """统一校准入口：读 ``plan.intent.calib``，供 LC / GPTQ / msmodelslim / … 共用。

    Recipe 示例::

        calib:
          source: open-perfectblend          # 包内默认目录
          # path: /data/calib/open-perfectblend  # 或自定义 save_to_disk 目录
          max_samples: 512
          batch_size: 1
          shuffle: true
          seed: 42
    """

    plan: BackendPlan
    forward_fn: Callable[[dict[str, Any]], None] | None = None

    def _calib(self) -> dict[str, Any]:
        return dict(self.plan.intent.calib or {})

    @property
    def source(self) -> str:
        return str(self._calib().get("source", "open-perfectblend"))

    @property
    def path(self) -> str | None:
        """recipe ``calib.path`` / ``calib.dir`` / ``calib.dataset_dir``。"""
        c = self._calib()
        for key in ("path", "dir", "dataset_dir"):
            if c.get(key):
                return str(c[key])
        return None

    @property
    def max_samples(self) -> int:
        return int(self._calib().get("max_samples", 512))

    @property
    def batch_size(self) -> int:
        return int(self._calib().get("batch_size", 1))

    @property
    def split(self) -> str:
        return str(self._calib().get("split", "train"))

    @property
    def text_column(self) -> str:
        return str(self._calib().get("text_column", "text"))

    @property
    def shuffle(self) -> bool:
        return bool(self._calib().get("shuffle", True))

    @property
    def seed(self) -> int:
        return int(self._calib().get("seed", 42))

    @property
    def max_seq_length(self) -> int:
        """tokenize / oneshot 序列长度上限（GPTQ 关键，默认 2048）。"""
        c = self._calib()
        return int(c.get("max_seq_length") or c.get("max_length") or 2048)

    @property
    def max_chars(self) -> int | None:
        """文本字符粗截断；默认 ≈ max_seq_length * 4，防超长对话撑爆内存。"""
        c = self._calib()
        if "max_chars" in c:
            v = c.get("max_chars")
            return int(v) if v is not None else None
        # ~4 chars/token 经验值；留一点余量
        return int(self.max_seq_length * 4)

    def world_size(self) -> int:
        return int(self._calib().get("world_size", 1))

    def resolved_dir(self) -> Path | None:
        return resolve_calib_path(self.source, path=self.path)

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        return iter_text_samples(
            self.source,
            max_samples=self.max_samples,
            split=self.split,
            text_column=self.text_column,
            path=self.path,
            shuffle=self.shuffle,
            seed=self.seed,
            max_chars=self.max_chars,
        )

    def iter_batches(self) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for sample in self.iter_samples():
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def load(self) -> CalibData:
        """加载并物化校准样本（各 backend 消费 ``texts`` / ``samples``）。"""
        if self._calib().get("distributed") and self.world_size() > 1:
            report = {
                "mode": "distributed_stub",
                "world_size": self.world_size(),
                "source": self.source,
                "path": self.path,
                "max_samples": self.max_samples,
            }
            logger.info("calib distributed stub world_size=%s", self.world_size())
            return CalibData(samples=[], texts=[], report=report)

        resolved = self.resolved_dir()
        samples = list(self.iter_samples())
        texts = [str(s.get("text", "")) for s in samples]
        report = {
            "mode": "local",
            "source": self.source,
            "path": str(resolved) if resolved else self.path,
            "default_perfectblend": str(default_perfectblend_dir()),
            "samples": len(samples),
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "max_seq_length": self.max_seq_length,
            "max_chars": self.max_chars,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }
        logger.info(
            "calib loaded source=%s path=%s samples=%d batch_size=%d "
            "max_seq_length=%d max_chars=%s",
            self.source,
            report.get("path"),
            len(samples),
            self.batch_size,
            self.max_seq_length,
            self.max_chars,
        )
        return CalibData(
            samples=samples,
            texts=texts,
            report=report,
            resolved_path=str(resolved) if resolved else None,
        )

    def run(self) -> dict[str, Any]:
        """加载样本；若设置 ``forward_fn`` 则逐条前向，返回 report。"""
        data = self.load()
        if self.forward_fn and data.samples:
            for batch in self.iter_batches():
                for item in batch:
                    self.forward_fn(item)
            data.report["forward_applied"] = True
        return data.report

    def oneshot_kwargs(self, data: CalibData | None = None) -> dict[str, Any]:
        """给 llmcompressor.oneshot 用的 dataset / num_calibration_samples / max_seq_length。

        优先传已物化的 HF Dataset（含 ``text`` 列），避免 oneshot 再去拉 hub。
        """
        calib = self._calib()
        out: dict[str, Any] = {
            "num_calibration_samples": int(calib.get("max_samples") or self.max_samples),
            "max_seq_length": self.max_seq_length,
        }

        data = data or self.load()
        if data.texts:
            try:
                from datasets import Dataset

                out["dataset"] = Dataset.from_dict({"text": data.texts})
                out["text_column"] = "text"
                return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("calib Dataset.from_dict failed: %s", exc)

        if data.resolved_path:
            out["dataset"] = data.resolved_path
        elif self.source and self.source != "stub":
            out["dataset"] = self.source
        return out

    def write_jsonl(self, path: Path, data: CalibData | None = None) -> Path:
        """写出 Ascend / msModelSlim 常用的 jsonl（每行含 ``text``）。"""
        data = data or self.load()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for sample in data.samples:
                row = (
                    {"text": sample.get("text", "")}
                    if "text" in sample
                    else dict(sample)
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("calib wrote jsonl path=%s samples=%d", path, len(data.samples))
        return path

    def resolve_calib_file(
        self,
        output_dir: Path,
        data: CalibData | None = None,
        *,
        explicit: str | Path | None = None,
    ) -> Path | None:
        """解析或物化校准文件：显式路径 > source 文件 > ``luban_calib.jsonl``。"""
        if explicit:
            return Path(explicit)
        if self.path:
            p = Path(self.path)
            if p.is_file():
                return p
        src = self.source
        if src and src != "stub":
            p = Path(src)
            if p.is_file():
                return p
        data = data or self.load()
        if not data.samples:
            return None
        return self.write_jsonl(Path(output_dir) / "luban_calib.jsonl", data)
