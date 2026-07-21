from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator

from result_companion.core.chunking.chunking import ChunkingStrategy, RenderLine
from result_companion.core.plugins.base import ParseOptions, TestChunkPayload

JUNIT_ROOT_TAGS = {"testsuite", "testsuites"}


@dataclass(frozen=True)
class PytestJUnitCase:
    """Single testcase parsed from pytest JUnit XML."""

    name: str
    classname: str
    status: str
    elapsed_time: str
    message: str
    stdout: str
    stderr: str


class PytestJUnitResults:
    """Parsed pytest JUnit XML results for Result Companion analysis."""

    def __init__(
        self,
        path: Path,
        cases: list[PytestJUnitCase],
        options: ParseOptions,
    ) -> None:
        self._path = path
        self._cases = cases
        self._options = options
        self._chunking: ChunkingStrategy | None = None

    @property
    def has_chunking(self) -> bool:
        """Returns True when a chunking strategy has been attached."""
        return self._chunking is not None

    def set_chunking(self, strategy: ChunkingStrategy) -> "PytestJUnitResults":
        """Attaches the current core chunking strategy."""
        self._chunking = strategy
        return self

    @property
    def total_test_count(self) -> int:
        """Returns testcase count before exclude_passing is applied."""
        return len(self._cases)

    @cached_property
    def test_names(self) -> list[str]:
        """Returns testcase names selected for analysis."""
        return [case.name for case in self._selected_cases()]

    @cached_property
    def source_hash(self) -> str:
        """Returns short SHA-256 hash of source bytes and parse options."""
        hasher = hashlib.sha256()
        with self._path.open("rb") as source:
            for chunk in iter(lambda: source.read(65_536), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
        hasher.update(_options_hash_bytes(self._options))
        return hasher.hexdigest()[:12]

    def render_chunks(self) -> Iterator[TestChunkPayload]:
        """Yields chunked payloads for selected testcases."""
        if self._chunking is None:
            raise ValueError("Call set_chunking() before render_chunks().")

        for case in self._selected_cases():
            lines = _render_case(case, self._options.exclude_fields or [])
            chunks, chunk_stats = self._chunking.apply(lines)
            yield TestChunkPayload(case.name, chunks, chunk_stats, case.status)

    def _selected_cases(self) -> Iterator[PytestJUnitCase]:
        """Yields cases selected by current parse options."""
        for case in self._cases:
            if self._options.exclude_passing and case.status in ("PASS", "SKIP"):
                continue
            yield case


class PytestJUnitPlugin:
    """Parses pytest JUnit XML artifacts."""

    name = "pytest-junit"

    def can_parse(self, path: Path) -> bool:
        """Returns True when the XML root looks like JUnit XML."""
        if path.suffix.lower() != ".xml":
            return False

        try:
            with path.open("rb") as source:
                _event, root = next(ET.iterparse(source, events=("start",)))
        except (ET.ParseError, StopIteration, OSError):
            return False
        return _local_name(root.tag) in JUNIT_ROOT_TAGS

    def parse(self, path: Path, options: ParseOptions) -> PytestJUnitResults:
        """Parses pytest JUnit XML into Result Companion results."""
        if options.include_tags or options.exclude_tags:
            raise ValueError("Format 'pytest-junit' does not support tag filters.")

        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise ValueError(f"pytest-junit could not parse {path}: {exc}") from exc

        if _local_name(root.tag) not in JUNIT_ROOT_TAGS:
            raise ValueError(f"pytest-junit could not parse {path}: not JUnit XML.")

        cases = [_parse_case(element) for element in root.iter() if _is_case(element)]
        return PytestJUnitResults(path=path, cases=cases, options=options)


def _parse_case(element: ET.Element) -> PytestJUnitCase:
    """Parses one testcase element."""
    status, message = _case_status_and_message(element)
    return PytestJUnitCase(
        name=_case_name(element),
        classname=element.attrib.get("classname", ""),
        status=status,
        elapsed_time=element.attrib.get("time", ""),
        message=message,
        stdout=_child_text(element, "system-out"),
        stderr=_child_text(element, "system-err"),
    )


def _case_status_and_message(element: ET.Element) -> tuple[str, str]:
    """Returns Result Companion status and diagnostic message."""
    for child in element:
        tag = _local_name(child.tag)
        if tag == "failure":
            return "FAIL", _diagnostic_text(child)
        if tag == "error":
            return "ERROR", _diagnostic_text(child)
        if tag == "skipped":
            return "SKIP", _diagnostic_text(child)
    return "PASS", ""


def _render_case(case: PytestJUnitCase, excluded_fields: list[str]) -> list[RenderLine]:
    """Renders one testcase as flat lines for the current chunking contract."""
    excluded = set(excluded_fields)
    lines = [RenderLine(0, f"Test: {case.name}")]
    _append_line(lines, excluded, "classname", case.classname)
    _append_line(lines, excluded, "status", case.status)
    _append_line(lines, excluded, "elapsed_time", case.elapsed_time)
    _append_line(lines, excluded, "message", case.message)
    _append_line(lines, excluded, "stdout", case.stdout)
    _append_line(lines, excluded, "stderr", case.stderr)
    return lines


def _append_line(
    lines: list[RenderLine],
    excluded_fields: set[str],
    field: str,
    value: str,
) -> None:
    """Appends a rendered field when present and not excluded."""
    if field in excluded_fields or not value:
        return
    lines.append(RenderLine(1, f"{field}: {value}"))


def _case_name(element: ET.Element) -> str:
    """Returns stable testcase display name."""
    name = element.attrib.get("name", "<unnamed>")
    classname = element.attrib.get("classname", "")
    if not classname:
        return name
    return f"{classname}.{name}"


def _diagnostic_text(element: ET.Element) -> str:
    """Combines JUnit diagnostic attributes and text."""
    parts = [
        element.attrib.get("type", ""),
        element.attrib.get("message", ""),
        (element.text or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def _child_text(element: ET.Element, child_name: str) -> str:
    """Returns text for a direct child by local tag name."""
    for child in element:
        if _local_name(child.tag) == child_name:
            return (child.text or "").strip()
    return ""


def _is_case(element: ET.Element) -> bool:
    """Returns True for testcase elements."""
    return _local_name(element.tag) == "testcase"


def _local_name(tag: str) -> str:
    """Strips XML namespace from tag name."""
    return tag.rsplit("}", 1)[-1]


def _options_hash_bytes(options: ParseOptions) -> bytes:
    """Serializes render-affecting parse options for source hashing."""
    payload = {
        "exclude_fields": sorted(options.exclude_fields or []),
        "exclude_passing": options.exclude_passing,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
