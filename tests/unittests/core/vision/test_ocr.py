import pytest

from result_companion.core.vision import ocr
from result_companion.core.vision.models import EmbeddedImage
from result_companion.core.vision.ocr import (
    OCR_INSTALL_HINT,
    OcrDependencies,
    run_ocr_batch,
)


class FakeOcrResult:
    def __init__(self, texts: list[str]) -> None:
        self.txts = texts


class FakeEngine:
    def __init__(self, results: list[object]) -> None:
        self.results = list(results)
        self.calls: list[object] = []

    def __call__(self, image: object) -> object:
        self.calls.append(image)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeImage:
    def __enter__(self) -> "FakeImage":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def convert(self, mode: str) -> str:
        return f"image:{mode}"


class FakeImageModule:
    @staticmethod
    def open(_image_bytes: object) -> FakeImage:
        return FakeImage()


class FakeNumpyModule:
    @staticmethod
    def array(image: object) -> object:
        return image


def make_image(
    image_id: str = "image-1",
    test_name: str = "Test",
    ordinal: int = 1,
) -> EmbeddedImage:
    return EmbeddedImage(
        id=image_id,
        test_name=test_name,
        keyword_path=("Capture Page Screenshot",),
        message_index=0,
        image_index=0,
        ordinal=ordinal,
        mime_type="image/png",
        data_base64="aGVsbG8=",
    )


def set_fake_dependencies(monkeypatch: pytest.MonkeyPatch, engine: FakeEngine) -> None:
    dependencies = OcrDependencies(
        engine=engine,
        image_module=FakeImageModule,
        numpy_module=FakeNumpyModule,
    )
    monkeypatch.setattr(ocr, "_load_ocr_dependencies", lambda: dependencies)


@pytest.mark.asyncio
async def test_run_ocr_batch_returns_text_by_image_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine([FakeOcrResult(["Login", "Password"])])
    set_fake_dependencies(monkeypatch, engine)

    result = await run_ocr_batch([make_image()], 3, 1500, 1)

    assert result == {"image-1": "Login\nPassword"}
    assert engine.calls == ["image:RGB"]


@pytest.mark.asyncio
async def test_run_ocr_batch_caps_images_per_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine([FakeOcrResult(["one"]), FakeOcrResult(["two"])])
    set_fake_dependencies(monkeypatch, engine)
    images = [
        make_image("image-1", test_name="Same", ordinal=1),
        make_image("image-2", test_name="Same", ordinal=2),
        make_image("image-3", test_name="Same", ordinal=3),
    ]

    result = await run_ocr_batch(images, 2, 1500, 1)

    assert result == {"image-1": "one", "image-2": "two"}
    assert len(engine.calls) == 2


@pytest.mark.asyncio
async def test_run_ocr_batch_truncates_text(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = FakeEngine([FakeOcrResult(["abcdef"])])
    set_fake_dependencies(monkeypatch, engine)

    result = await run_ocr_batch([make_image()], 3, 3, 1)

    assert result == {"image-1": "abc"}


@pytest.mark.asyncio
async def test_run_ocr_batch_skips_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = FakeEngine([FakeOcrResult(["  "])])
    set_fake_dependencies(monkeypatch, engine)

    result = await run_ocr_batch([make_image()], 3, 1500, 1)

    assert result == {}


@pytest.mark.asyncio
async def test_run_ocr_batch_raises_install_hint_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependencies() -> OcrDependencies:
        raise RuntimeError(OCR_INSTALL_HINT)

    monkeypatch.setattr(ocr, "_load_ocr_dependencies", missing_dependencies)

    with pytest.raises(RuntimeError, match="result-companion\\[ocr\\]"):
        await run_ocr_batch([make_image()], 3, 1500, 1)


@pytest.mark.asyncio
async def test_run_ocr_batch_continues_when_one_image_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine([ValueError("boom"), FakeOcrResult(["ok"])])
    set_fake_dependencies(monkeypatch, engine)
    images = [make_image("bad"), make_image("good")]

    result = await run_ocr_batch(images, 3, 1500, 1)

    assert result == {"good": "ok"}


def test_extract_text_accepts_list_result_shape() -> None:
    result = ocr._extract_text([["box", "First", 0.9], ["box", "Second", 0.8]])

    assert result == "First\nSecond"
