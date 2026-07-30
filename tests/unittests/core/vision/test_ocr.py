import sys
from types import ModuleType

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
    data_base64: str = "aGVsbG8=",
    test_identity: tuple[str, ...] | None = None,
) -> EmbeddedImage:
    return EmbeddedImage(
        id=image_id,
        test_name=test_name,
        test_identity=test_identity or ("Suite", test_name),
        keyword_path=("Capture Page Screenshot",),
        message_index=0,
        image_index=0,
        ordinal=ordinal,
        mime_type="image/png",
        data_base64=data_base64,
    )


def set_fake_dependencies(monkeypatch: pytest.MonkeyPatch, engine: FakeEngine) -> None:
    dependencies = OcrDependencies(
        engine=engine,
        image_module=FakeImageModule,
        numpy_module=FakeNumpyModule,
    )
    monkeypatch.setattr(ocr, "_load_ocr_dependencies", lambda: dependencies)


def test_load_ocr_dependencies_uses_optional_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_numpy = ModuleType("numpy")
    fake_pil = ModuleType("PIL")
    fake_image = ModuleType("PIL.Image")
    fake_rapidocr = ModuleType("rapidocr")
    engines: list[object] = []

    class FakeRapidOCR:
        def __init__(self) -> None:
            engines.append(self)

    fake_pil.Image = fake_image
    fake_rapidocr.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)
    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr)

    dependencies = ocr._load_ocr_dependencies()

    assert dependencies.engine is engines[0]
    assert dependencies.image_module is fake_image
    assert dependencies.numpy_module is fake_numpy


def test_load_ocr_dependencies_raises_install_hint_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "rapidocr", None)

    with pytest.raises(RuntimeError, match="result-companion\\[ocr\\]"):
        ocr._load_ocr_dependencies()


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
async def test_run_ocr_batch_returns_empty_when_limit_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ocr,
        "_load_ocr_dependencies",
        lambda: pytest.fail("OCR dependencies should not load"),
    )

    result = await run_ocr_batch([make_image()], 0, 1500, 1)

    assert result == {}


@pytest.mark.asyncio
async def test_run_ocr_batch_caps_images_per_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine([FakeOcrResult(["one"]), FakeOcrResult(["two"])])
    set_fake_dependencies(monkeypatch, engine)
    images = [
        make_image("image-1", test_name="Same", ordinal=1, data_base64="b25l"),
        make_image("image-2", test_name="Same", ordinal=2, data_base64="dHdv"),
        make_image("image-3", test_name="Same", ordinal=3, data_base64="dGhyZWU="),
    ]

    result = await run_ocr_batch(images, 2, 1500, 1)

    assert result == {"image-1": "one", "image-2": "two"}
    assert len(engine.calls) == 2


@pytest.mark.asyncio
async def test_run_ocr_batch_caps_duplicate_test_names_by_test_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine([FakeOcrResult(["one"]), FakeOcrResult(["three"])])
    set_fake_dependencies(monkeypatch, engine)
    images = [
        make_image("image-1", "Same", 1, "b25l", ("Root", "A", "Same")),
        make_image("image-2", "Same", 2, "dHdv", ("Root", "A", "Same")),
        make_image("image-3", "Same", 1, "dGhyZWU=", ("Root", "B", "Same")),
        make_image("image-4", "Same", 2, "Zm91cg==", ("Root", "B", "Same")),
    ]

    result = await run_ocr_batch(images, 1, 1500, 1)

    assert result == {"image-1": "one", "image-3": "three"}
    assert len(engine.calls) == 2


@pytest.mark.asyncio
async def test_run_ocr_batch_reuses_text_for_duplicate_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeEngine([FakeOcrResult(["same"])])
    set_fake_dependencies(monkeypatch, engine)
    images = [
        make_image("image-1", test_name="First"),
        make_image("image-2", test_name="Second"),
    ]

    result = await run_ocr_batch(images, 3, 1500, 1)

    assert result == {"image-1": "same", "image-2": "same"}
    assert len(engine.calls) == 1


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
    images = [make_image("bad", data_base64="YmFk"), make_image("good")]

    result = await run_ocr_batch(images, 3, 1500, 1)

    assert result == {"good": "ok"}


def test_extract_text_accepts_list_result_shape() -> None:
    result = ocr._extract_text([["box", "First", 0.9], ["box", "Second", 0.8]])

    assert result == "First\nSecond"


def test_extract_text_accepts_tuple_result_shape() -> None:
    result = ocr._extract_text(([["box", "Tuple", 0.9]], "meta"))

    assert result == "Tuple"


def test_extract_text_returns_empty_for_unknown_result() -> None:
    assert ocr._extract_text(object()) == ""


def test_text_from_item_extracts_string_variants() -> None:
    assert ocr._text_from_item("direct") == "direct"
    assert ocr._text_from_item((123, None, "fallback")) == "fallback"
    assert ocr._text_from_item(object()) == ""


def test_truncate_text_returns_empty_for_non_positive_limit() -> None:
    assert ocr._truncate_text("abc", 0) == ""
