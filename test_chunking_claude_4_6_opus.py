#!/usr/bin/env python3
"""Structure-aware chunking demo for Robot Framework output.xml.

Usage:
    python test_chunking_claude_4_6_opus.py output.xml [--chunk-size 500] [--fail-only]
"""
import argparse
from pathlib import Path

from robot.api import ExecutionResult

INDENT = "    "
SKIP_LEVELS = {"TRACE", "DEBUG"}


def _elapsed(item) -> str:
    """Formats elapsed time from an RF model item."""
    et = getattr(item, "elapsed_time", None)
    if not et:
        return ""
    secs = et.total_seconds()
    if secs >= 1:
        return f" ({secs:.1f}s)"
    if secs >= 0.001:
        return f" ({secs * 1000:.0f}ms)"
    return ""


def render_body(body: list, depth: int, lines: list[tuple[int, str]]) -> None:
    """Recursively renders keyword/message body items as (depth, text) pairs."""
    for item in body:
        if getattr(item, "type", None) == "MESSAGE":
            if item.level not in SKIP_LEVELS:
                lines.append((depth, f"[{item.level}] {item.message}"))
            continue
        if not hasattr(item, "body"):
            continue
        name = getattr(item, "name", "") or getattr(item, "type", "?")
        status = getattr(item, "status", "")
        kind = (
            item.type.title()
            if getattr(item, "type", "") in ("SETUP", "TEARDOWN")
            else "Keyword"
        )
        lines.append((depth, f"{kind}: {name} - {status}{_elapsed(item)}"))
        if getattr(item, "args", None):
            lines.append((depth + 1, f"args: {', '.join(str(a) for a in item.args)}"))
        assign = getattr(item, "var", None) or getattr(item, "assign", None)
        if assign:
            lines.append((depth + 1, f"=> {', '.join(str(a) for a in assign)}"))
        render_body(item.body, depth + 1, lines)


def render_test(test, suite_names: list[tuple[str, str]]) -> list[tuple[int, str]]:
    """Renders a test case with suite ancestry as (depth, text) lines."""
    lines: list[tuple[int, str]] = []
    for i, (name, status) in enumerate(suite_names):
        lines.append((i, f"Suite: {name} - {status}"))
    d = len(suite_names)
    lines.append((d, f"Test: {test.name} - {test.status}{_elapsed(test)}"))
    render_body(test.body, d + 1, lines)
    return lines


def walk_tests(suite, path: list[tuple[str, str]] | None = None):
    """Yields (test, suite_path) for every test in the suite tree."""
    if path is None:
        path = []
    current = path + [(suite.name, suite.status)]
    for test in suite.tests:
        yield test, current
    for child in suite.suites:
        yield from walk_tests(child, current)


def _breadcrumb_at(lines: list[tuple[int, str]], at_idx: int) -> list[tuple[int, str]]:
    """Walks backwards to collect one ancestor per depth level above at_idx."""
    target = lines[at_idx][0] - 1
    ancestors: list[tuple[int, str]] = []
    for i in range(at_idx - 1, -1, -1):
        if lines[i][0] == target:
            ancestors.insert(0, lines[i])
            target -= 1
            if target < 0:
                break
    return ancestors


def _fmt(depth: int, text: str) -> str:
    return f"{INDENT * depth}{text}"


def chunk_lines(lines: list[tuple[int, str]], chunk_size: int) -> list[str]:
    """Splits rendered lines into chunks, prepending breadcrumb context to each."""
    if not lines:
        return []

    total = sum(len(_fmt(d, t)) + 1 for d, t in lines)
    if total <= chunk_size:
        return ["\n".join(_fmt(d, t) for d, t in lines)]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for idx, (depth, text) in enumerate(lines):
        fmt = _fmt(depth, text)
        line_len = len(fmt) + 1

        if line_len > chunk_size:
            bc = _breadcrumb_at(lines, idx)
            bc_lines = [_fmt(d, t) for d, t in bc]
            bc_size = sum(len(l) + 1 for l in bc_lines) + len(_fmt(depth, "{...}")) + 1
            prefix_len = len(INDENT) * depth
            remaining = text

            if current:
                space_left = chunk_size - current_size - prefix_len
                if space_left > 0:
                    current.append(_fmt(depth, remaining[:space_left]))
                    remaining = remaining[space_left:]
                chunks.append("\n".join(current))
                current, current_size = [], 0

            avail = max(chunk_size - bc_size - prefix_len, chunk_size // 3)
            pieces = [remaining[i : i + avail] for i in range(0, len(remaining), avail)]

            for piece in pieces[:-1]:
                chunk_body = bc_lines + [_fmt(depth, "{...}"), _fmt(depth, piece)]
                chunks.append("\n".join(chunk_body))

            if pieces:
                current = bc_lines + [_fmt(depth, "{...}"), _fmt(depth, pieces[-1])]
                current_size = sum(len(l) + 1 for l in current)
            continue

        if current_size + line_len > chunk_size and current:
            chunks.append("\n".join(current))
            bc = _breadcrumb_at(lines, idx)
            current = [_fmt(d, t) for d, t in bc]
            current.append(_fmt(depth, "{...}"))
            current_size = sum(len(l) + 1 for l in current)

        current.append(fmt)
        current_size += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure-aware chunking demo")
    parser.add_argument("xml_path", type=Path, help="Path to RF output.xml")
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Max chars per chunk"
    )
    parser.add_argument(
        "--fail-only", action="store_true", help="Only show failing tests"
    )
    args = parser.parse_args()

    result = ExecutionResult(str(args.xml_path))

    for test, suite_path in walk_tests(result.suite):
        if args.fail_only and test.status == "PASS":
            continue

        lines = render_test(test, suite_path)
        full_text = "\n".join(_fmt(d, t) for d, t in lines)

        print(f"\n{'=' * 70}")
        print(f"TEST: {test.name}  |  Full rendered: {len(full_text)} chars")
        print(f"{'=' * 70}")

        chunks = chunk_lines(lines, args.chunk_size)

        if len(chunks) == 1:
            print("\n[Single chunk — no splitting needed]\n")
            print(chunks[0])
        else:
            for i, chunk in enumerate(chunks):
                print(f"\n--- Chunk {i + 1}/{len(chunks)} (len={len(chunk)}) ---\n")
                print(chunk)

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
