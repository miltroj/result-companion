import pytest

import result_companion.core.utils.llm_debug as llm_debug
from result_companion.core.utils.llm_debug import (
    enable_llm_debug,
    is_llm_debug_enabled,
    write_llm_record,
)


@pytest.fixture(autouse=True)
def reset_writer():
    llm_debug._writer = None
    yield
    llm_debug._writer = None


def test_is_llm_debug_enabled_false_by_default():
    assert not is_llm_debug_enabled()


def test_is_llm_debug_enabled_true_after_enable(tmp_path):
    enable_llm_debug(tmp_path / "debug.log")
    assert is_llm_debug_enabled()


def test_enable_llm_debug_writes_record(tmp_path):
    path = tmp_path / "debug.log"
    enable_llm_debug(path)

    write_llm_record(
        label="[My Test] (single chunk)", prompt="my prompt", response="my response"
    )

    content = path.read_text(encoding="utf-8")
    assert "[My Test] (single chunk)" in content
    assert "my prompt" in content
    assert "my response" in content


def test_write_llm_record_appends_multiple_records(tmp_path):
    path = tmp_path / "debug.log"
    enable_llm_debug(path)

    write_llm_record(label="[Test A]", prompt="prompt A", response="response A")
    write_llm_record(label="[Test B]", prompt="prompt B", response="response B")

    content = path.read_text(encoding="utf-8")
    assert "prompt A" in content
    assert "prompt B" in content
