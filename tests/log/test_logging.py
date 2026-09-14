"""Logging format and level tests (luban_sculpt.log.logging)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from luban_sculpt.log import (
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILENAME,
    LOG_LEVEL_NAMES,
    get_log_file_path,
    get_logger,
    resolve_log_dir,
    resolve_log_level,
    set_log_level,
)
from luban_sculpt.log.logging import DEFAULT_LOG_FORMAT


def test_log_format_includes_module_func_line() -> None:
    buf = io.StringIO()
    set_log_level(level="debug", stream=buf, log_dir="", force=True)
    log = get_logger("luban_sculpt.test_logging")

    log.info("hello pipeline")

    line = buf.getvalue().strip()
    assert "luban_sculpt.test_logging" in line
    assert "test_log_format_includes_module_func_line:" in line
    assert "| INFO | hello pipeline" in line
    assert line[0:4].isdigit()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warn", logging.WARNING),
        ("error", logging.ERROR),
    ],
)
def test_resolve_log_level_names(name: str, expected: int) -> None:
    assert resolve_log_level(name) == expected


def test_four_log_levels_emit_expected_tags() -> None:
    buf = io.StringIO()
    set_log_level(level="debug", stream=buf, log_dir="", force=True)
    log = get_logger("luban_sculpt.test_levels")

    log.debug("d")
    log.info("i")
    log.warning("w")
    log.error("e")

    text = buf.getvalue()
    assert "| DEBUG | d" in text
    assert "| INFO | i" in text
    assert "| WARNING | w" in text
    assert "| ERROR | e" in text


def test_info_level_filters_debug() -> None:
    buf = io.StringIO()
    set_log_level(level="info", stream=buf, log_dir="", force=True)
    log = get_logger("luban_sculpt.test_filter")
    log.debug("hidden")
    log.info("visible")
    text = buf.getvalue()
    assert "hidden" not in text
    assert "| INFO | visible" in text


def test_error_level_only_error_and_above() -> None:
    buf = io.StringIO()
    set_log_level(level="error", stream=buf, log_dir="", force=True)
    log = get_logger("luban_sculpt.test_filter")
    log.info("hidden")
    log.error("shown")
    text = buf.getvalue()
    assert "hidden" not in text
    assert "| ERROR | shown" in text


def test_log_level_names_are_four_cli_choices() -> None:
    assert LOG_LEVEL_NAMES == ("debug", "info", "warn", "error")


def test_default_log_format_unchanged() -> None:
    assert "funcName" in DEFAULT_LOG_FORMAT
    assert "%(name)s" in DEFAULT_LOG_FORMAT


def test_modules_use_get_logger_api() -> None:
    from luban_sculpt.backends.base import logger as base_logger
    from luban_sculpt.modifiers.builtins import logger as builtins_logger
    from luban_sculpt.model.resolve import logger as resolve_logger

    for log in (base_logger, builtins_logger, resolve_logger):
        assert isinstance(log, logging.Logger)
        assert log.name.startswith("luban_sculpt.")


def test_resolve_log_dir_defaults_to_local_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LUBAN_LOG_DIR", raising=False)
    assert resolve_log_dir() == Path(DEFAULT_LOG_DIR)
    assert resolve_log_dir(None) == Path(DEFAULT_LOG_DIR)


def test_resolve_log_dir_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUBAN_LOG_DIR", "/from/env")
    assert resolve_log_dir("/explicit", fallback="/fb") == Path("/explicit")
    assert resolve_log_dir(None, fallback="/fb") == Path("/from/env")
    monkeypatch.delenv("LUBAN_LOG_DIR", raising=False)
    assert resolve_log_dir(None, fallback="/fb") == Path("/fb")
    assert resolve_log_dir(None, fallback=None) == Path(DEFAULT_LOG_DIR)


def test_set_log_level_writes_file(tmp_path: Path) -> None:
    buf = io.StringIO()
    path = set_log_level(
        level="info",
        stream=buf,
        log_dir=tmp_path,
        force=True,
    )
    assert path is not None
    assert path.name == DEFAULT_LOG_FILENAME
    assert path.parent == tmp_path.resolve()
    assert get_log_file_path() == path

    log = get_logger("luban_sculpt.test_file")
    log.info("to file and stream")

    stream_text = buf.getvalue()
    assert "to file and stream" in stream_text

    file_text = path.read_text(encoding="utf-8")
    assert "to file and stream" in file_text
    assert "luban_sculpt.test_file" in file_text


def test_set_log_level_default_local_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LUBAN_LOG_DIR", raising=False)
    buf = io.StringIO()
    path = set_log_level(level="info", stream=buf, force=True)
    assert path is not None
    assert path == (tmp_path / DEFAULT_LOG_DIR / DEFAULT_LOG_FILENAME).resolve()
    assert path.is_file()


def test_set_log_level_empty_log_dir_disables_file() -> None:
    buf = io.StringIO()
    path = set_log_level(level="info", stream=buf, log_dir="", force=True)
    assert path is None
    assert get_log_file_path() is None
