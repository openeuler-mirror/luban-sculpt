"""Quant backend protocol and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Type

from luban_sculpt.contracts import BackendPlan, QuantizedArtifact
from luban_sculpt.log import get_logger

logger = get_logger(__name__)


class QuantBackend(ABC):
    """量化后端插件：接收 BackendPlan，落盘权重与 manifest。"""

    name: str = "base"

    @abstractmethod
    def quantize(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        """执行量化；output_dir 为产物根目录。"""
        raise NotImplementedError


def _discover_backends() -> dict[str, Type[QuantBackend]]:
    """通过 entry_points 组 ``luban_sculpt.backends`` 发现已安装后端。

    单个 entry 加载失败时跳过并 warning（便于扩展包半成品不阻断主路径）。
    """
    mapping: dict[str, Type[QuantBackend]] = {}
    try:
        eps = entry_points(group="luban_sculpt.backends")
    except TypeError:
        eps = entry_points().get("luban_sculpt.backends", [])
    for ep in eps:
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001 — plugin isolation
            logger.warning("skip backend entry %r: %s", ep.name, exc)
            continue
        mapping[ep.name] = cls
    return mapping


class BackendRegistry:
    """已注册后端名称 → 实例工厂。"""

    def __init__(self) -> None:
        self._backends = _discover_backends()

    def get(self, name: str) -> QuantBackend:
        if name not in self._backends:
            msg = f"Unknown backend {name!r}; registered: {list(self._backends)}"
            logger.error(msg)
            raise KeyError(msg)
        return self._backends[name]()

    def registered(self) -> list[str]:
        return sorted(self._backends.keys())


class BackendRouter:
    """按 plan.intent.backend 路由到对应 QuantBackend。"""

    def __init__(self, registry: BackendRegistry | None = None) -> None:
        self.registry = registry or BackendRegistry()

    def run(self, plan: BackendPlan, output_dir: str) -> QuantizedArtifact:
        """调用后端 quantize。"""
        logger.info(
            "backend route name=%s scheme=%s export=%s output=%s",
            plan.intent.backend,
            plan.intent.abstract_scheme,
            plan.export_format.value,
            output_dir,
        )
        try:
            backend = self.registry.get(plan.intent.backend)
            artifact = backend.quantize(plan, output_dir)
        except Exception:
            logger.error(
                "backend quantize failed name=%s output=%s",
                plan.intent.backend,
                output_dir,
                exc_info=True,
            )
            raise
        logger.info(
            "backend done name=%s output=%s",
            plan.intent.backend,
            artifact.output_dir,
        )
        return artifact
