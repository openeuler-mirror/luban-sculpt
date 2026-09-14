"""Logging configuration implementation."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TextIO

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_FILENAME = "luban-sculpt.log"
# 相对当前工作目录的默认本地日志目录
DEFAULT_LOG_DIR = "logs"

LOG_LEVEL_NAMES: tuple[str, ...] = ("debug", "info", "warn", "error")

_LEVEL_BY_NAME: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_configured = False
_log_file_path: Path | None = None


def resolve_log_level(level: int | str) -> int:
    """Parse ``debug`` / ``info`` / ``warn`` / ``error`` or a ``logging.*`` int."""
    if isinstance(level, int):
        return level
    key = str(level).strip().lower()
    if key not in _LEVEL_BY_NAME:
        valid = ", ".join(LOG_LEVEL_NAMES)
        raise ValueError(f"Unknown log level {level!r}; use one of: {valid}")
    return _LEVEL_BY_NAME[key]


def resolve_log_dir(
    explicit: str | Path | None = None,
    *,
    fallback: str | Path | None = None,
) -> Path:
    """解析日志目录。

    优先级：``explicit`` → 环境变量 ``LUBAN_LOG_DIR`` → ``fallback`` → ``DEFAULT_LOG_DIR``（``./logs``）。
    """
    if explicit is not None and str(explicit).strip():
        return Path(str(explicit).strip())
    env = os.environ.get("LUBAN_LOG_DIR")
    if env and env.strip():
        return Path(env.strip())
    if fallback is not None and str(fallback).strip():
        return Path(str(fallback).strip())
    return Path(DEFAULT_LOG_DIR)


def get_log_file_path() -> Path | None:
    """当前进程已打开的日志文件路径（未启用文件日志则为 None）。"""
    return _log_file_path


def set_log_level(
    *,
    level: int | str = "info",
    stream: TextIO | None = None,
    log_dir: str | Path | None = None,
    log_filename: str = DEFAULT_LOG_FILENAME,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    force: bool = False,
) -> Path | None:
    """设置日志 level，并写入 ``log_dir/log_filename``（默认 ``./logs``）。

    - ``log_dir`` 为 None：使用 :func:`resolve_log_dir`（含默认本地 ``logs/``）
    - ``log_dir`` 为 ``""``：仅 stderr，不写文件（测试用）

    始终挂 stderr（或 ``stream``）StreamHandler。
    返回打开的日志文件路径（未写文件则为 None）。
    """
    global _configured, _log_file_path
    if _configured and not force:
        return _log_file_path

    resolved = resolve_log_level(level)
    formatter = logging.Formatter(log_format, datefmt=date_format)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved)

    stream_handler = logging.StreamHandler(
        stream if stream is not None else sys.stderr
    )
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(resolved)
    root.addHandler(stream_handler)

    _log_file_path = None
    # 空字符串显式关闭文件日志；None 走默认本地目录
    if log_dir != "":
        out_dir = resolve_log_dir(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / log_filename
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(resolved)
        root.addHandler(file_handler)
        _log_file_path = path.resolve()

    _configured = True
    return _log_file_path


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; call :func:`set_log_level` at process entry if not yet done."""
    return logging.getLogger(name)
