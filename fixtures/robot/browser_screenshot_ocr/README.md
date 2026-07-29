# Browser Screenshot OCR Harness

## Quick Read

This Robot Framework fixture opens a wrong `result-companion.com` path, embeds a Browser screenshot, then fails on missing text. Use it to generate `output.xml` with an inline screenshot for Result Companion vision checks.

## Setup

Install Browser only when running this manual harness:

```bash
poetry run pip install robotframework-browser
poetry run rfbrowser init
```

## Run

From the repository root, generate Robot output:

```bash
poetry run robot --outputdir .rc-browser-harness fixtures/robot/browser_screenshot_ocr
```

Check Result Companion sees the embedded screenshot placeholder:

```bash
poetry run python fixtures/robot/browser_screenshot_ocr/dump_llm_texts.py
```

This writes `.rc-browser-harness/llm_texts.txt`. Each test starts with a `===== TEST: ... =====` separator. Text after that separator is direct `results.as_texts()` output after the same field filtering and vision config used before LLM calls.

Expected text contains:

```text
[SCREENSHOT] embedded image/png screenshot #1
```

Use custom paths when needed:

```bash
poetry run python fixtures/robot/browser_screenshot_ocr/dump_llm_texts.py \
  .rc-browser-harness/output.xml \
  .rc-browser-harness/manual_check.txt
```

Run the CLI with vision placeholders enabled:

```bash
poetry run result-companion analyze \
  -o .rc-browser-harness/output.xml \
  -c fixtures/robot/browser_screenshot_ocr/vision_enabled.yaml \
  --dryrun
```
