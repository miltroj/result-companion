from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

from robot.api import ExecutionResult
from robot.errors import DataError
from robot.result.model import Keyword, Message, TestCase, TestSuite

from result_companion.core.chunking.chunking import (
    ChunkingStrategy,
    RenderLine,
    deduplicate_consecutive_lines,
    render_lines_to_text,
)
from result_companion.core.chunking.utils import Chunking
from result_companion.core.results.visitors import UniqueNameResultVisitor
from result_companion.core.utils.logging_config import get_progress_logger

logger = get_progress_logger("RFResults")

ALL_FIELDS = frozenset(
    {
        "name",
        "status",
        "type",
        "args",
        "message",
        "doc",
        "tags",
        "setup",
        "teardown",
        "level",
        "elapsed_time",
        "assign",
        "timestamp",
        "lineno",
        "owner",
    }
)


class RenderedTest(NamedTuple):
    """Internal representation of a test with its render context."""

    name: str
    status: str
    lines: list[RenderLine]


@dataclass
class TestLines:
    """Rendered lines for a single test with suite ancestry context."""

    name: str
    lines: list[RenderLine]

    def __str__(self) -> str:
        return render_lines_to_text(self.lines)


class ContextAwareRobotResults:
    """Iterates RF result tree per-test with suite context, field filtering, and chunking.

    Per-test iteration (no chunking)::

        for test_name, text in results.as_texts():
            send_to_llm(text)

    Per-test with token-aware chunking::

        strategy = ChunkingStrategy(tokenizer_config=tokenizer, system_prompt=prompt)
        for test_name, chunks, chunk_stats, test_status in results.set_chunking(strategy).render_chunks():
            for chunk in chunks:
                send_to_llm(chunk)
    """

    def __init__(self, source: ExecutionResult | Path | TestSuite) -> None:
        if isinstance(source, (str, Path)):
            self._result = ExecutionResult(source)
            self._result.visit(UniqueNameResultVisitor())
            self._suite = self._result.suite
        elif isinstance(source, TestSuite):
            self._result = None
            self._suite = source
        else:
            self._result = source
            self._suite = source.suite
        self._fields: frozenset[str] = ALL_FIELDS
        self._chunking: ChunkingStrategy | None = None
        self._exclude_passing: bool = False

    def _invalidate_cache(self) -> None:
        for attr in ("test_names", "source_hash"):
            self.__dict__.pop(attr, None)

    def include_fields(self, fields: Sequence[str]) -> ContextAwareRobotResults:
        """Sets which fields to render (replaces all)."""
        self._fields = frozenset(fields)
        self._invalidate_cache()
        return self

    def exclude_fields(self, fields: Sequence[str]) -> ContextAwareRobotResults:
        """Removes fields from active set."""
        self._fields = self._fields - frozenset(fields)
        self._invalidate_cache()
        return self

    def include_tags(self, tags: Sequence[str]) -> ContextAwareRobotResults:
        """Filters to tests matching any of given tags (RF native, supports wildcards)."""
        self._apply_config({"include_tags": list(tags)})
        return self

    def exclude_tags(self, tags: Sequence[str]) -> ContextAwareRobotResults:
        """Excludes tests matching any of given tags (RF native, supports wildcards)."""
        self._apply_config({"exclude_tags": list(tags)})
        return self

    def _apply_config(self, suite_config: dict) -> None:
        if self._result is None:
            raise TypeError(
                "Source is TestSuite, not ExecutionResult, TAG filtering is not available!"
            )
        try:
            self._result.configure(suite_config=suite_config)
        except DataError as exc:
            raise ValueError(
                f"Tag filter {suite_config} matched zero tests. "
                f"Check include_tags/exclude_tags in your config or CLI args. "
                f"RF error: {exc}"
            ) from exc
        self._suite = self._result.suite
        self._invalidate_cache()

    def exclude_passing(self, exclude: bool = True) -> ContextAwareRobotResults:
        """When True, skips tests with PASS or SKIP status from iteration."""
        self._exclude_passing = exclude
        self._invalidate_cache()
        return self

    @property
    def has_chunking(self) -> bool:
        """True if a chunking strategy has been set."""
        return self._chunking is not None

    def set_chunking(self, strategy: ChunkingStrategy) -> ContextAwareRobotResults:
        """Attaches a chunking strategy."""
        self._chunking = strategy
        return self

    def _iter_tests(self) -> Iterator[RenderedTest]:
        """Internal iterator with passing-test filter and line deduplication applied."""
        for name, status, lines in _iter_tests_with_context(
            self._suite, [], 0, self._fields
        ):
            if self._exclude_passing and status in ("PASS", "SKIP"):
                continue
            yield RenderedTest(name, status, deduplicate_consecutive_lines(lines))

    def __iter__(self) -> Iterator[tuple[str, TestLines]]:
        """Yields (test_name, TestLines) for each test with suite ancestry context."""
        for name, _status, lines in self._iter_tests():
            yield name, TestLines(name=name, lines=lines)

    def as_texts(self) -> Iterator[tuple[str, str]]:
        """Yields (test_name, rendered_text) — each test as LLM-ready string."""
        for name, test_lines in self:
            yield name, str(test_lines)

    @property
    def total_test_count(self) -> int:
        """Total tests in (tag-filtered) suite, ignoring exclude_passing."""
        return sum(1 for _ in self._suite.all_tests)

    @cached_property
    def source_hash(self) -> str:
        """Short SHA-256 hash of the rendered suite for reproducibility tracking."""
        blob = str(self).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @cached_property
    def test_names(self) -> list[str]:
        """Names of tests that pass current filters (excluding passing if set)."""
        return [name for name, _, _ in self._iter_tests()]

    def __str__(self) -> str:
        return render_lines_to_text(_render_suite(self._suite, 0, self._fields))

    def render_chunks(self) -> Iterator[tuple[str, list[str], Chunking, str]]:
        """Yields (test_name, chunks, chunk_stats, test_status) per test.

        Raises:
            ValueError: If no ChunkingStrategy has been set.
        """
        if self._chunking is None:
            raise ValueError("Call set_chunking() before render_chunks().")
        for test_name, test_status, lines in self._iter_tests():
            chunks, chunk_stats = self._chunking.apply(lines)
            yield test_name, chunks, chunk_stats, test_status or "N/A"


def get_rc_robot_results(
    file_path: Path,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    exclude_fields: Sequence[str] | None = None,
    exclude_passing: bool = True,
    chunking_strategy: ChunkingStrategy | None = None,
) -> ContextAwareRobotResults:
    """Facade: parses output.xml and returns a configured ContextAwareRobotResults.

    Args:
        file_path: Path to RF output.xml.
        include_tags: RF tag patterns to include (supports wildcards).
        exclude_tags: RF tag patterns to exclude (supports wildcards).
        exclude_fields: Field names to omit from rendering.
        exclude_passing: When True, skips tests with PASS status.
        chunking_strategy: Optional token-aware chunking for render_chunks().
    """
    results = ContextAwareRobotResults(file_path)
    if include_tags:
        results.include_tags(include_tags)
    if exclude_tags:
        results.exclude_tags(exclude_tags)
    if exclude_fields:
        results.exclude_fields(exclude_fields)
    if exclude_passing:
        results.exclude_passing()
    if chunking_strategy:
        results.set_chunking(chunking_strategy)
    return results


def _iter_tests_with_context(
    suite: TestSuite,
    ancestor_lines: list[RenderLine],
    depth: int,
    fields: frozenset[str],
    ancestor_teardowns: list[RenderLine] | None = None,
) -> Iterator[RenderedTest]:
    """Yields RenderedTest for each test, with ancestor suite context prepended."""
    if ancestor_teardowns is None:
        ancestor_teardowns = []
    context = ancestor_lines + (
        [RenderLine(depth, f"Suite: {suite.name}")] if "name" in fields else []
    )
    if suite.has_setup:
        if "setup" in fields:
            context = context + _render_keyword(suite.setup, depth + 1, fields)
        if getattr(suite.setup, "status", None) == "FAIL":
            skipped = sum(1 for _ in suite.all_tests)
            logger.warning(
                f"Suite setup FAILED for '{suite.name}' — "
                f"collapsing {skipped} skipped test(s) into single analysis unit."
            )
            suite_teardown = (
                _render_keyword(suite.teardown, depth + 1, fields)
                if "teardown" in fields and suite.has_teardown
                else []
            )
            yield RenderedTest(
                suite.name, "FAIL", context + suite_teardown + ancestor_teardowns
            )
            return
    suite_teardown = (
        _render_keyword(suite.teardown, depth + 1, fields)
        if "teardown" in fields and suite.has_teardown
        else []
    )
    all_teardowns = suite_teardown + ancestor_teardowns
    for test in suite.tests:
        yield RenderedTest(
            test.name,
            test.status,
            context + _render_test(test, depth + 1, fields) + all_teardowns,
        )
    for child in suite.suites:
        yield from _iter_tests_with_context(
            child, context, depth + 1, fields, all_teardowns
        )


def _render_suite(
    suite: TestSuite, depth: int, fields: frozenset[str]
) -> list[RenderLine]:
    """Recursively renders a suite and its children."""
    lines: list[RenderLine] = []
    if "name" in fields:
        lines.append(RenderLine(depth, f"Suite: {suite.name}"))
    if "doc" in fields and suite.doc:
        lines.append(RenderLine(depth + 1, f"doc: {suite.doc}"))
    if "setup" in fields and suite.has_setup:
        lines.extend(_render_keyword(suite.setup, depth + 1, fields))
    for test in suite.tests:
        lines.extend(_render_test(test, depth + 1, fields))
    for child in suite.suites:
        lines.extend(_render_suite(child, depth + 1, fields))
    if "teardown" in fields and suite.has_teardown:
        lines.extend(_render_keyword(suite.teardown, depth + 1, fields))
    return lines


def _render_common_fields(
    obj: TestCase | Keyword, depth: int, fields: frozenset[str]
) -> list[RenderLine]:
    """Renders elapsed_time, lineno, doc, tags — shared by test and keyword."""
    lines: list[RenderLine] = []
    if "elapsed_time" in fields and obj.elapsed_time is not None:
        lines.append(RenderLine(depth, f"elapsed: {obj.elapsed_time}"))
    if "lineno" in fields and getattr(obj, "lineno", None):
        lines.append(RenderLine(depth, f"lineno: {obj.lineno}"))
    if "doc" in fields and obj.doc:
        lines.append(RenderLine(depth, f"doc: {obj.doc}"))
    if "tags" in fields and obj.tags:
        lines.append(RenderLine(depth, f"tags: {', '.join(obj.tags)}"))
    return lines


def _render_test(
    test: TestCase, depth: int, fields: frozenset[str]
) -> list[RenderLine]:
    """Renders a test case header and its body."""
    header = _join_parts(
        test.name if "name" in fields else None,
        test.status if "status" in fields else None,
    )
    lines: list[RenderLine] = [RenderLine(depth, f"Test: {header}")]
    lines.extend(_render_common_fields(test, depth + 1, fields))
    if "owner" in fields and getattr(test, "owner", None):
        lines.append(RenderLine(depth + 1, f"owner: {test.owner}"))
    if "setup" in fields and test.has_setup:
        lines.extend(_render_keyword(test.setup, depth + 1, fields))
    for item in test.body:
        if getattr(item, "type", "").lower() in ("setup", "teardown"):
            continue
        lines.extend(_render_body_item(item, depth + 1, fields))
    if "teardown" in fields and test.has_teardown:
        lines.extend(_render_keyword(test.teardown, depth + 1, fields))
    return lines


def _render_keyword(
    kw: Keyword, depth: int, fields: frozenset[str]
) -> list[RenderLine]:
    """Renders a keyword header, args, and its body recursively."""
    kind = kw.type.title() if "type" in fields else "Keyword"
    header = _join_parts(
        kw.name if "name" in fields else None,
        kw.status if "status" in fields else None,
    )
    lines: list[RenderLine] = [RenderLine(depth, f"{kind}: {header}")]
    if "args" in fields and kw.args:
        lines.append(
            RenderLine(depth + 1, f"args: {', '.join(str(a) for a in kw.args)}")
        )
    if "assign" in fields and kw.assign:
        lines.append(RenderLine(depth + 1, f"assign: {', '.join(kw.assign)}"))
    lines.extend(_render_common_fields(kw, depth + 1, fields))
    for item in kw.body:
        lines.extend(_render_body_item(item, depth + 1, fields))
    return lines


def _render_body_item(
    item: object, depth: int, fields: frozenset[str]
) -> list[RenderLine]:
    """Dispatches rendering: Keyword, Message, or recurses into control structures."""
    if isinstance(item, Message):
        return _render_message(item, depth, fields)
    if isinstance(item, Keyword):
        return _render_keyword(item, depth, fields)
    body = getattr(item, "body", None)
    if not body:
        return []
    lines: list[RenderLine] = []
    for child in body:
        lines.extend(_render_body_item(child, depth, fields))
    return lines


def _render_message(
    msg: Message, depth: int, fields: frozenset[str]
) -> list[RenderLine]:
    """Renders a log message, optionally prefixed with level."""
    if "message" not in fields:
        return []
    prefix = ""
    if "timestamp" in fields and msg.timestamp:
        prefix += f"{msg.timestamp} "
    if "level" in fields:
        prefix += f"[{msg.level}] "
    return [RenderLine(depth, f"{prefix}{msg.message}")]


def _join_parts(*parts: str | None) -> str:
    """Joins non-None parts with ' - '."""
    return " - ".join(p for p in parts if p)
