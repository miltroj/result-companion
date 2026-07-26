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
from result_companion.core.vision.extractor import scan_html_images, strip_html_images
from result_companion.core.vision.models import EmbeddedImage

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


@dataclass
class RenderContext:
    """Mutable Robot render context for embedded screenshot correlation."""

    suite_path: tuple[str, ...]
    test_name: str
    keyword_path: tuple[str, ...] = ()
    message_index: int = 0
    image_ordinal: int = 0


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

    def __init__(self, source: ExecutionResult | str | Path | TestSuite) -> None:
        self._source_path: Path | None = None
        if isinstance(source, (str, Path)):
            self._source_path = Path(source)
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
        self._include_images: bool = False
        self._image_texts: dict[str, str] = {}
        self._fallback_source_hash = (
            None if self._source_path else _hash_rendered_suite(self._suite)
        )

    def _invalidate_cache(self) -> None:
        for attr in ("test_names",):
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

    def include_embedded_images(self, include: bool = True) -> ContextAwareRobotResults:
        """Renders embedded screenshot placeholders inline."""
        self._include_images = include
        self._invalidate_cache()
        return self

    def collect_embedded_images(self) -> list[EmbeddedImage]:
        """Returns embedded images from currently selected tests."""
        images: list[EmbeddedImage] = []
        for rendered_test, test_images in _iter_tests_with_context_and_images(
            self._suite,
            [],
            0,
            self._fields,
        ):
            if self._exclude_passing and rendered_test.status in ("PASS", "SKIP"):
                continue
            images.extend(test_images)
        return images

    def attach_image_texts(self, texts: dict[str, str]) -> ContextAwareRobotResults:
        """Attaches OCR text by ``EmbeddedImage.id`` and enables image rendering."""
        self._image_texts = dict(texts)
        self._include_images = True
        self._invalidate_cache()
        return self

    def _iter_tests(self) -> Iterator[RenderedTest]:
        """Internal iterator with passing-test filter and line deduplication applied."""
        include_images = self._include_images or bool(self._image_texts)
        for name, status, lines in _iter_tests_with_context(
            self._suite,
            [],
            0,
            self._fields,
            include_images=include_images,
            image_texts=self._image_texts,
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
        """Short SHA-256 hash of the raw source identity."""
        # TODO: Add analysis_hash for analyzed-result-set identity. source_hash only
        # tracks raw output.xml identity; analysis_hash should combine source_hash,
        # selected tests, tag/pass filters, field exclusions, and vision/OCR config so
        # reports from the same source but different analysis scope do not collide.
        if self._source_path:
            return _hash_file(self._source_path)
        return self._fallback_source_hash or _hash_rendered_suite(self._suite)

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


def _hash_file(path: Path) -> str:
    """Returns short SHA-256 hash for raw file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _hash_rendered_suite(suite: TestSuite) -> str:
    """Returns short SHA-256 hash for rendered suite fallback."""
    text = render_lines_to_text(_render_suite(suite, 0, ALL_FIELDS))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _render_suite_teardown(
    suite: TestSuite,
    depth: int,
    fields: frozenset[str],
    context: RenderContext | None = None,
    include_images: bool = False,
    image_texts: dict[str, str] | None = None,
    collected_images: list[EmbeddedImage] | None = None,
) -> list[RenderLine]:
    """Renders suite teardown if field enabled and teardown exists."""
    if "teardown" not in fields or not suite.has_teardown:
        return []
    return _render_keyword(
        suite.teardown,
        depth + 1,
        fields,
        context=context,
        include_images=include_images,
        image_texts=image_texts,
        collected_images=collected_images,
    )


def _iter_tests_with_context(
    suite: TestSuite,
    ancestor_lines: list[RenderLine],
    depth: int,
    fields: frozenset[str],
    ancestor_teardowns: list[RenderLine] | None = None,
    include_images: bool = False,
    image_texts: dict[str, str] | None = None,
) -> Iterator[RenderedTest]:
    """Yields RenderedTest for each test, with ancestor suite context prepended."""
    for rendered_test, _images in _iter_tests_with_context_and_images(
        suite,
        ancestor_lines,
        depth,
        fields,
        ancestor_teardowns,
        include_images,
        image_texts,
    ):
        yield rendered_test


def _iter_tests_with_context_and_images(
    suite: TestSuite,
    ancestor_lines: list[RenderLine],
    depth: int,
    fields: frozenset[str],
    ancestor_teardowns: list[RenderLine] | None = None,
    include_images: bool = False,
    image_texts: dict[str, str] | None = None,
    suite_path: tuple[str, ...] = (),
) -> Iterator[tuple[RenderedTest, list[EmbeddedImage]]]:
    """Yields rendered tests plus images collected for each rendered test."""
    if ancestor_teardowns is None:
        ancestor_teardowns = []

    current_suite_path = suite_path + (suite.name,)
    base_context = ancestor_lines + (
        [RenderLine(depth, f"Suite: {suite.name}")] if "name" in fields else []
    )
    context = base_context
    if suite.has_setup and "setup" in fields:
        context = context + _render_keyword(suite.setup, depth + 1, fields)

    suite_teardown = _render_suite_teardown(suite, depth, fields)

    if suite.has_setup and getattr(suite.setup, "status", None) == "FAIL":
        skipped = sum(1 for _ in suite.all_tests)
        logger.warning(
            f"Suite setup FAILED for '{suite.name}' — "
            f"collapsing {skipped} skipped test(s) into single analysis unit."
        )
        images: list[EmbeddedImage] = []
        render_context = RenderContext(
            suite_path=current_suite_path,
            test_name=suite.name,
        )
        collapsed_context = list(base_context)
        if "setup" in fields:
            collapsed_context.extend(
                _render_keyword(
                    suite.setup,
                    depth + 1,
                    fields,
                    context=render_context,
                    include_images=include_images,
                    image_texts=image_texts,
                    collected_images=images,
                )
            )
        collapsed_teardowns = _render_suite_teardown(
            suite,
            depth,
            fields,
            context=render_context,
            include_images=include_images,
            image_texts=image_texts,
            collected_images=images,
        )
        yield (
            RenderedTest(
                suite.name,
                "FAIL",
                collapsed_context + collapsed_teardowns + ancestor_teardowns,
            ),
            images,
        )
        return

    all_teardowns = suite_teardown + ancestor_teardowns
    for test in suite.tests:
        images = []
        render_context = RenderContext(
            suite_path=current_suite_path,
            test_name=test.name,
        )
        test_context = list(base_context)
        if suite.has_setup and "setup" in fields:
            test_context.extend(
                _render_keyword(
                    suite.setup,
                    depth + 1,
                    fields,
                    context=render_context,
                    include_images=include_images,
                    image_texts=image_texts,
                    collected_images=images,
                )
            )
        test_teardowns = _render_suite_teardown(
            suite,
            depth,
            fields,
            context=render_context,
            include_images=include_images,
            image_texts=image_texts,
            collected_images=images,
        )
        yield (
            RenderedTest(
                test.name,
                test.status,
                test_context
                + _render_test(
                    test,
                    depth + 1,
                    fields,
                    context=render_context,
                    include_images=include_images,
                    image_texts=image_texts,
                    collected_images=images,
                )
                + test_teardowns
                + ancestor_teardowns,
            ),
            images,
        )
    for child in suite.suites:
        yield from _iter_tests_with_context_and_images(
            child,
            context,
            depth + 1,
            fields,
            all_teardowns,
            include_images,
            image_texts,
            current_suite_path,
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
    test: TestCase,
    depth: int,
    fields: frozenset[str],
    context: RenderContext | None = None,
    include_images: bool = False,
    image_texts: dict[str, str] | None = None,
    collected_images: list[EmbeddedImage] | None = None,
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
        lines.extend(
            _render_keyword(
                test.setup,
                depth + 1,
                fields,
                context=context,
                include_images=include_images,
                image_texts=image_texts,
                collected_images=collected_images,
            )
        )
    for item in test.body:
        if getattr(item, "type", "").lower() in ("setup", "teardown"):
            continue
        lines.extend(
            _render_body_item(
                item,
                depth + 1,
                fields,
                context=context,
                include_images=include_images,
                image_texts=image_texts,
                collected_images=collected_images,
            )
        )
    if "teardown" in fields and test.has_teardown:
        lines.extend(
            _render_keyword(
                test.teardown,
                depth + 1,
                fields,
                context=context,
                include_images=include_images,
                image_texts=image_texts,
                collected_images=collected_images,
            )
        )
    return lines


def _render_keyword(
    kw: Keyword,
    depth: int,
    fields: frozenset[str],
    context: RenderContext | None = None,
    include_images: bool = False,
    image_texts: dict[str, str] | None = None,
    collected_images: list[EmbeddedImage] | None = None,
) -> list[RenderLine]:
    """Renders a keyword header, args, and its body recursively."""
    old_keyword_path = None
    if context is not None:
        old_keyword_path = context.keyword_path
        keyword_name = kw.name or getattr(kw, "type", "Keyword")
        context.keyword_path = old_keyword_path + (keyword_name,)

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
        lines.extend(
            _render_body_item(
                item,
                depth + 1,
                fields,
                context=context,
                include_images=include_images,
                image_texts=image_texts,
                collected_images=collected_images,
            )
        )
    if context is not None and old_keyword_path is not None:
        context.keyword_path = old_keyword_path
    return lines


def _render_body_item(
    item: object,
    depth: int,
    fields: frozenset[str],
    context: RenderContext | None = None,
    include_images: bool = False,
    image_texts: dict[str, str] | None = None,
    collected_images: list[EmbeddedImage] | None = None,
) -> list[RenderLine]:
    """Dispatches rendering: Keyword, Message, or recurses into control structures."""
    if isinstance(item, Message):
        return _render_message(
            item,
            depth,
            fields,
            context=context,
            include_images=include_images,
            image_texts=image_texts,
            collected_images=collected_images,
        )
    if isinstance(item, Keyword):
        return _render_keyword(
            item,
            depth,
            fields,
            context=context,
            include_images=include_images,
            image_texts=image_texts,
            collected_images=collected_images,
        )
    body = getattr(item, "body", None)
    if not body:
        return []
    lines: list[RenderLine] = []
    old_keyword_path = None
    if context is not None:
        old_keyword_path = context.keyword_path
        segment = _control_path_segment(item)
        if segment:
            context.keyword_path = old_keyword_path + (segment,)
    for child in body:
        lines.extend(
            _render_body_item(
                child,
                depth,
                fields,
                context=context,
                include_images=include_images,
                image_texts=image_texts,
                collected_images=collected_images,
            )
        )
    if context is not None and old_keyword_path is not None:
        context.keyword_path = old_keyword_path
    return lines


def _render_message(
    msg: Message,
    depth: int,
    fields: frozenset[str],
    context: RenderContext | None = None,
    include_images: bool = False,
    image_texts: dict[str, str] | None = None,
    collected_images: list[EmbeddedImage] | None = None,
) -> list[RenderLine]:
    """Renders a log message, optionally prefixed with level."""
    if "message" not in fields:
        return []
    prefix = ""
    if "timestamp" in fields and msg.timestamp:
        prefix += f"{msg.timestamp} "
    if "level" in fields:
        prefix += f"[{msg.level}] "
    # Embedded data URI images are stripped even when placeholders are disabled.
    message = strip_html_images(str(msg.message or ""))
    lines = [RenderLine(depth, f"{prefix}{message}")] if message.strip() else []
    if context is None:
        return lines

    scanned_images = scan_html_images(str(msg.message or ""))
    if not scanned_images:
        context.message_index += 1
        return lines

    message_index = context.message_index
    context.message_index += 1
    for image_index, (mime_type, data_base64) in enumerate(scanned_images):
        context.image_ordinal += 1
        image = _embedded_image(
            context=context,
            message_index=message_index,
            image_index=image_index,
            ordinal=context.image_ordinal,
            mime_type=mime_type,
            data_base64=data_base64,
        )
        if collected_images is not None:
            collected_images.append(image)
        if include_images:
            lines.extend(_render_image_event(image, depth, image_texts or {}))
    return lines


def _embedded_image(
    context: RenderContext,
    message_index: int,
    image_index: int,
    ordinal: int,
    mime_type: str,
    data_base64: str,
) -> EmbeddedImage:
    """Builds stable image metadata from render traversal context."""
    data_hash = hashlib.sha256(data_base64.encode()).hexdigest()[:12]
    key_parts = (
        *context.suite_path,
        context.test_name,
        *context.keyword_path,
        str(message_index),
        str(image_index),
        data_hash,
    )
    image_id = hashlib.sha256("\0".join(key_parts).encode()).hexdigest()[:24]
    return EmbeddedImage(
        id=image_id,
        test_name=context.test_name,
        keyword_path=context.keyword_path,
        message_index=message_index,
        image_index=image_index,
        ordinal=ordinal,
        mime_type=mime_type,
        data_base64=data_base64,
    )


def _render_image_event(
    image: EmbeddedImage,
    depth: int,
    image_texts: dict[str, str],
) -> list[RenderLine]:
    """Renders a screenshot placeholder and attached OCR text lines."""
    lines = [RenderLine(depth, image.placeholder())]
    text = image_texts.get(image.id, "")
    lines.extend(
        RenderLine(depth, f"[SCREENSHOT_OCR] {line.strip()}")
        for line in text.splitlines()
        if line.strip()
    )
    return lines


def _control_path_segment(item: object) -> str | None:
    """Returns a stable-ish path segment for RF control structures."""
    item_type = getattr(item, "type", None)
    name = getattr(item, "name", None)
    if item_type and name:
        return f"{item_type}:{name}"
    return item_type or name


def _join_parts(*parts: str | None) -> str:
    """Joins non-None parts with ' - '."""
    return " - ".join(p for p in parts if p)
