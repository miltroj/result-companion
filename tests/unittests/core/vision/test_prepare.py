import pytest

from result_companion.core.parsers.config import DefaultConfigModel
from result_companion.core.vision import prepare
from result_companion.core.vision.models import EmbeddedImage
from result_companion.core.vision.prepare import prepare_vision_results


class FakeResults:
    def __init__(self, images: list[EmbeddedImage] | None = None) -> None:
        self.images = images or []
        self.include_calls = 0
        self.collect_calls = 0
        self.attached_texts: dict[str, str] | None = None

    def include_embedded_images(self) -> "FakeResults":
        self.include_calls += 1
        return self

    def collect_embedded_images(self) -> list[EmbeddedImage]:
        self.collect_calls += 1
        return self.images

    def attach_image_texts(self, texts: dict[str, str]) -> "FakeResults":
        self.attached_texts = texts
        return self


def make_config() -> DefaultConfigModel:
    return DefaultConfigModel(
        version=1.0,
        llm_config={
            "question_prompt": "question prompt",
            "prompt_template": "template {question} {context}",
            "chunking": {
                "chunk_analysis_prompt": "Analyze: {text}",
                "final_synthesis_prompt": "Synthesize: {summary}",
            },
            "summary_prompt_template": "CI summary:\n{analyses}",
        },
        llm_factory={"model": "openai/gpt-4"},
        tokenizer={"tokenizer": "openai_tokenizer", "max_content_tokens": 1000},
    )


def make_image() -> EmbeddedImage:
    return EmbeddedImage(
        id="image-1",
        test_name="Test",
        keyword_path=("Capture Page Screenshot",),
        message_index=0,
        image_index=0,
        ordinal=1,
        mime_type="image/png",
        data_base64="aGVsbG8=",
    )


@pytest.mark.asyncio
async def test_prepare_vision_results_enables_placeholders_only() -> None:
    config = make_config()
    config.vision.enabled = True
    results = FakeResults()

    prepared = await prepare_vision_results(results, config)

    assert prepared is results
    assert results.include_calls == 1
    assert results.collect_calls == 0


@pytest.mark.asyncio
async def test_prepare_vision_results_skips_ocr_during_dryrun() -> None:
    config = make_config()
    config.vision.ocr = True
    results = FakeResults([make_image()])

    await prepare_vision_results(results, config, dryrun=True)

    assert results.include_calls == 1
    assert results.collect_calls == 0
    assert results.attached_texts is None


@pytest.mark.asyncio
async def test_prepare_vision_results_logs_when_ocr_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    debug_messages: list[str] = []
    monkeypatch.setattr(prepare.logger, "debug", debug_messages.append)
    config = make_config()
    config.vision.ocr = True

    await prepare_vision_results(FakeResults(), config, dryrun=True)

    assert debug_messages == ["OCR enabled for embedded screenshots."]


@pytest.mark.asyncio
async def test_prepare_vision_results_attaches_ocr_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ocr(**kw: object) -> dict[str, str]:
        return {"image-1": "Login"}

    monkeypatch.setattr(prepare, "run_ocr_batch", fake_ocr)
    config = make_config()
    config.vision.ocr = True
    results = FakeResults([make_image()])

    await prepare_vision_results(results, config)

    assert results.include_calls == 1
    assert results.collect_calls == 1
    assert results.attached_texts == {"image-1": "Login"}
