# pytest-junit Parser Plugin Example

Minimal non-Robot parser plugin for pytest JUnit XML.

Use it as a canary for the current plugin contract:

```bash
pip install -e examples/plugins/pytest_junit
result-companion analyze -o junit.xml --format pytest-junit --no-html-report --text-report rc.txt --dryrun
```

This example intentionally implements `set_chunking()` because the current `ParsedResults`
contract requires it, even though flat JUnit test cases do not need Robot Framework breadcrumb
chunking.
