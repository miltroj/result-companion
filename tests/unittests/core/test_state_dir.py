"""Tests for durable user state directory."""

from pathlib import Path

from result_companion.core.state_dir import get_or_create_current_user_rc_state_dir


def test_get_user_state_dir_creates_directory_under_home(tmp_path: Path) -> None:
    state = get_or_create_current_user_rc_state_dir(home=tmp_path)

    assert state == tmp_path / ".result-companion"
    assert state.is_dir()


def test_get_user_state_dir_second_call_reuses_existing(tmp_path: Path) -> None:
    first = get_or_create_current_user_rc_state_dir(home=tmp_path)
    second = get_or_create_current_user_rc_state_dir(home=tmp_path)

    assert first == second
