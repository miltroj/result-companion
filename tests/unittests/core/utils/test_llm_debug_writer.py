from pathlib import Path

from result_companion.core.utils.logging_config import make_llm_debug_writer


def test_make_llm_debug_writer_appends_to_file(tmp_path: Path):
    path = tmp_path / "debug.log"
    writer = make_llm_debug_writer(path)

    writer("first record\n")
    writer("second record\n")

    content = path.read_text(encoding="utf-8")
    assert "first record" in content
    assert "second record" in content


def test_make_llm_debug_writer_creates_file_if_missing(tmp_path: Path):
    path = tmp_path / "new_debug.log"
    assert not path.exists()

    make_llm_debug_writer(path)("hello\n")

    assert path.exists()
    assert "hello" in path.read_text(encoding="utf-8")
