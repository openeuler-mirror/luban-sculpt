"""Logging format and level tests."""

from __future__ import annotations

import io
import logging

import pytest

from luban_sculpt.log import (
    LOG_LEVEL_NAMES,
    configure_logging,
    get_logger,
    resolve_log_level,
)


def test_log_format_includes_module_func_line() -> None:
    buf = io.StringIO()
    configure_logging(level="debug", stream=buf, force=True)
    log = get_logger("luban_sculpt.test_logging")

    log.info("hello pipeline")

    line = buf.getvalue().strip()
    assert "luban_sculpt.test_logging" in line
    assert "test_log_format_includes_module_func_line:" in line
    assert "| INFO | hello pipeline" in line
    assert line[0:4].isdigit()


@pytest.mark.parametrize("name,expected", [
    ("debug", logging.DEBUG),
    ("info", logging.INFO),
    ("warn", logging.WARNING),
    ("error", logging.ERROR),
])
def test_resolve_log_level_names(name: str, expected: int) -> None:
    assert resolve_log_level(name) == expected


def test_four_log_levels_emit_expected_tags() -> None:
    buf = io.StringIO()
    configure_logging(level="debug", stream=buf, force=True)
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
    configure_logging(level="info", stream=buf, force=True)
    log = get_logger("luban_sculpt.test_filter")
    log.debug("hidden")
    log.info("visible")
    text = buf.getvalue()
    assert "hidden" not in text
    assert "| INFO | visible" in text


def test_error_level_only_error_and_above() -> None:
    buf = io.StringIO()
    configure_logging(level="error", stream=buf, force=True)
    log = get_logger("luban_sculpt.test_filter")
    log.info("hidden")
    log.error("shown")
    text = buf.getvalue()
    assert "hidden" not in text
    assert "| ERROR | shown" in text


def test_log_level_names_are_four_cli_choices() -> None:
    assert LOG_LEVEL_NAMES == ("debug", "info", "warn", "error")
