"""Context-aware chunking built on RF native objects (Suite/Test/Keyword/Message)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from robot.api import ExecutionResult
from robot.result.model import Keyword, Message, TestCase, TestSuite

from result_companion.core.chunking.chunking import chunk_rf_test_lines

_DEFAULT_FIELDS = frozenset({"name", "status", "type", "args", "message"})


class ChunkableResult:
    """Walks RF native object tree with configurable field rendering.

    Usage:
        chunks = (
            ChunkableResult(path_or_suite)
            .include_fields(["name", "status", "args", "message"])
            .exclude_fields(["elapsed_time"])
            .for_chunk_size(4000)
        )
    """

    def __init__(self, source: Path | TestSuite) -> None:
        if isinstance(source, TestSuite):
            self._suite = source
        else:
            self._suite = ExecutionResult(source).suite
        self._include: frozenset[str] | None = None
        self._exclude: frozenset[str] = frozenset()

    def include_fields(self, fields: Sequence[str]) -> ChunkableResult:
        """Sets which fields to render (replaces defaults)."""
        self._include = frozenset(fields)
        return self

    def exclude_fields(self, fields: Sequence[str]) -> ChunkableResult:
        """Removes fields from the active set."""
        self._exclude = frozenset(fields)
        return self

    def _active_fields(self) -> frozenset[str]:
        base = self._include if self._include is not None else _DEFAULT_FIELDS
        return base - self._exclude

    def render_lines(self) -> list[tuple[int, str]]:
        """Renders the RF tree as (depth, text) pairs."""
        return _render_suite(self._suite, depth=0, fields=self._active_fields())

    def for_chunk_size(self, chunk_size: int) -> list[str]:
        """Renders lines then delegates to chunk_rf_test_lines."""
        return chunk_rf_test_lines(self.render_lines(), chunk_size)


def _render_suite(
    suite: TestSuite, depth: int, fields: frozenset[str]
) -> list[tuple[int, str]]:
    """Recursively renders a suite and its children."""
    lines: list[tuple[int, str]] = []
    if "name" in fields:
        lines.append((depth, f"Suite: {suite.name}"))
    if "doc" in fields and suite.doc:
        lines.append((depth + 1, f"doc: {suite.doc}"))
    if "setup" in fields and suite.has_setup:
        lines.extend(_render_keyword(suite.setup, depth + 1, fields))
    for test in suite.tests:
        lines.extend(_render_test(test, depth + 1, fields))
    for child in suite.suites:
        lines.extend(_render_suite(child, depth + 1, fields))
    if "teardown" in fields and suite.has_teardown:
        lines.extend(_render_keyword(suite.teardown, depth + 1, fields))
    return lines


def _render_test(
    test: TestCase, depth: int, fields: frozenset[str]
) -> list[tuple[int, str]]:
    """Renders a test case header and its body."""
    header = _join_parts(
        test.name if "name" in fields else None,
        test.status if "status" in fields else None,
    )
    lines: list[tuple[int, str]] = [(depth, f"Test: {header}")]
    if "tags" in fields and test.tags:
        lines.append((depth + 1, f"tags: {', '.join(test.tags)}"))
    if "doc" in fields and test.doc:
        lines.append((depth + 1, f"doc: {test.doc}"))
    for item in test.body:
        lines.extend(_render_body_item(item, depth + 1, fields))
    return lines


def _render_keyword(
    kw: Keyword, depth: int, fields: frozenset[str]
) -> list[tuple[int, str]]:
    """Renders a keyword header, args, and its body recursively."""
    kind = kw.type.title() if "type" in fields else "Keyword"
    header = _join_parts(
        kw.name if "name" in fields else None,
        kw.status if "status" in fields else None,
    )
    lines: list[tuple[int, str]] = [(depth, f"{kind}: {header}")]
    if "args" in fields and kw.args:
        lines.append((depth + 1, f"args: {', '.join(str(a) for a in kw.args)}"))
    if "doc" in fields and kw.doc:
        lines.append((depth + 1, f"doc: {kw.doc}"))
    if "tags" in fields and kw.tags:
        lines.append((depth + 1, f"tags: {', '.join(kw.tags)}"))
    for item in kw.body:
        lines.extend(_render_body_item(item, depth + 1, fields))
    return lines


def _render_body_item(
    item: Keyword | Message, depth: int, fields: frozenset[str]
) -> list[tuple[int, str]]:
    """Dispatches rendering based on item type."""
    if isinstance(item, Message):
        return _render_message(item, depth, fields)
    if isinstance(item, Keyword):
        return _render_keyword(item, depth, fields)
    return []


def _render_message(
    msg: Message, depth: int, fields: frozenset[str]
) -> list[tuple[int, str]]:
    """Renders a log message, optionally prefixed with level."""
    if "message" not in fields:
        return []
    text = f"[{msg.level}] {msg.message}" if "level" in fields else msg.message
    return [(depth, text)]


def _join_parts(*parts: str | None) -> str:
    """Joins non-None parts with ' - '."""
    return " - ".join(p for p in parts if p)
