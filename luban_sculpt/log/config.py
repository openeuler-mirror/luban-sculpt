"""Logging configuration implementation."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(name)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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


def resolve_log_level(level: int | str) -> int:
    """Parse ``debug`` / ``info`` / ``warn`` / ``error`` or a ``logging.*`` int."""
    if isinstance(level, int):
        return level
    key = str(level).strip().lower()
    if key not in _LEVEL_BY_NAME:
        valid = ", ".join(LOG_LEVEL_NAMES)
        raise ValueError(f"Unknown log level {level!r}; use one of: {valid}")
    return _LEVEL_BY_NAME[key]


def configure_logging(
    *,
    level: int | str = "info",
    stream: TextIO | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    force: bool = False,
) -> None:
    """Configure root logging once (safe to call from CLI / tests)."""
    global _configured
    if _configured and not force:
        return

    resolved = resolve_log_level(level)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; call :func:`configure_logging` at process entry if not yet done."""
    return logging.getLogger(name)
