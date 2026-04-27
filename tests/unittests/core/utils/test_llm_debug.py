from result_companion.core.utils.llm_debug import LLMDebugLogger


def test_disabled_by_default():
    assert not LLMDebugLogger().enabled


def test_enabled_after_from_path(tmp_path):
    logger = LLMDebugLogger.from_path(tmp_path / "debug.log")
    assert logger.enabled


def test_write_record_is_noop_when_disabled(tmp_path):
    logger = LLMDebugLogger()
    logger.write_record(label="test", prompt="p", response="r")


def test_write_record_appends_to_file(tmp_path):
    path = tmp_path / "debug.log"
    logger = LLMDebugLogger.from_path(path)

    logger.write_record(
        label="[My Test] (single chunk)", prompt="my prompt", response="my response"
    )

    content = path.read_text(encoding="utf-8")
    assert "[My Test] (single chunk)" in content
    assert "my prompt" in content
    assert "my response" in content


def test_write_record_appends_multiple_records(tmp_path):
    path = tmp_path / "debug.log"
    logger = LLMDebugLogger.from_path(path)

    logger.write_record(label="[Test A]", prompt="prompt A", response="response A")
    logger.write_record(label="[Test B]", prompt="prompt B", response="response B")

    content = path.read_text(encoding="utf-8")
    assert "prompt A" in content
    assert "prompt B" in content
