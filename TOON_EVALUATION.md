# TOON Format Evaluation for Result Companion

## Quick Read

TOON ([toonify](https://pypi.org/project/toonify/)) was evaluated as a drop-in replacement for `str(dict)` when serializing Robot Framework test cases for LLM input. Result: **not beneficial** for this data shape. Reverted.

---

## Context

Result Companion sends RF test case dicts to an LLM as text. The serialization path:

```
output.xml → result_parser (list[dict]) → str(dict) → LLM prompt context
```

TOON promises 40-60% token reduction for structured data by combining YAML-like indentation with CSV-style tabular arrays.

## Test Results

Same test case:

| Serialization | Raw length | Tokens (approx) |
|---------------|-----------|-----------------|
| `str(dict)` | 811,051 | ~278,513 |
| `toon.encode` | 822,093 | ~286,204 |
| **Difference** | **+1.4%** | **+2.8%** |

TOON produced **more** tokens, not fewer.

## Why It Failed

TOON saves tokens on **uniform arrays of objects** (identical keys, primitive values) by declaring headers once. RF test case dicts are the opposite:

- Deeply nested keyword trees (keywords within keywords)
- Non-uniform structures (different args, varying body depth)
- Mixed types at every level

TOON's own [benchmarks](https://github.com/toon-format/toon) confirm this:

| Data structure | TOON vs compact repr |
|---|---|
| Uniform arrays (100% tabular) | -36% tokens |
| Semi-uniform (~50% tabular) | +20% tokens |
| Deeply nested (0% tabular) | +12% tokens |

RF test data falls into the "deeply nested, non-uniform" category.

## Conclusion

Format-level changes won't reduce tokens for this data shape. `str(dict)` is already near-optimal for deeply nested non-uniform structures.

Effective token reduction strategies for this codebase:

- **Prune more fields** in [`result_parser.py`](result_companion/core/parsers/result_parser.py) before sending to LLM
- **Limit nesting depth** -- truncate keyword body trees beyond a useful depth
- **Truncate long values** -- cap string field lengths (e.g., log messages)
- **Structured extraction** -- flatten the keyword tree into a diagnostic summary tailored to LLM analysis needs

## Reference

- [TOON spec](https://github.com/toon-format/toon)
- [toonify (Python)](https://pypi.org/project/toonify/) -- v1.6.0, MIT
- [TOON benchmarks](https://github.com/toon-format/toon#benchmarks) -- accuracy and token efficiency data
