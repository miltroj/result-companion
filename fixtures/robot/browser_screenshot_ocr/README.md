# Browser Screenshot OCR Harness

## Quick Read

This Robot Framework fixture opens a local static page, embeds Browser screenshots, then fails on missing text. Use it to generate deterministic `output.xml` with inline screenshots for Result Companion vision checks.

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

This writes `.rc-browser-harness/llm_texts.txt`. Each test starts with a `===== TEST: ... =====` separator. Text after that separator is direct `results.as_texts()` output after the same field filtering and placeholder config used before LLM calls. The harness config excludes low-signal Browser docs, tags, timestamps, and local path metadata.

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

Run the CLI with vision placeholders:

```bash
poetry run result-companion analyze \
  -o .rc-browser-harness/output.xml \
  -c fixtures/robot/browser_screenshot_ocr/vision_enabled.yaml \
  --dryrun
```

Run with local OCR after installing the extra:

```bash
poetry install --extras ocr
poetry run result-companion analyze \
  -o .rc-browser-harness/output.xml \
  --ocr
```

OCR output adds `[SCREENSHOT_OCR]` lines when embedded screenshots contain readable text. OCR text is assistive and can be noisy, so do not treat it as assertion-grade truth. First OCR run may take longer while OCR dependencies initialize their local models.

File-based screenshots in `log.html` are not supported by this harness.

Committed `browser_self_contained/*.xml` files are frozen Browser-output examples and may not match the current harness exactly.
