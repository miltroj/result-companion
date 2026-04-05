"""Tests for durable user state directory."""

from pathlib import Path

from result_companion.core.state_dir import (
    APP,
    CLI,
    get_or_create_current_user_rc_state_dir,
    get_or_create_rc_state_subdirectory,
)


def test_get_user_state_dir_creates_directory_under_home(tmp_path: Path) -> None:
    state = get_or_create_current_user_rc_state_dir(home=tmp_path)

    assert state == tmp_path / ".result-companion"
    assert state.is_dir()


def test_get_user_state_dir_second_call_reuses_existing(tmp_path: Path) -> None:
    first = get_or_create_current_user_rc_state_dir(home=tmp_path)
    second = get_or_create_current_user_rc_state_dir(home=tmp_path)

    assert first == second


def test_get_or_create_rc_state_subdirectory_cli_and_data(tmp_path: Path) -> None:
    cli = get_or_create_rc_state_subdirectory(CLI, home=tmp_path)
    data = get_or_create_rc_state_subdirectory(CLI, "data", home=tmp_path)

    assert cli == tmp_path / ".result-companion" / "cli"
    assert data == cli / "data"
    assert cli.is_dir() and data.is_dir()


def test_get_or_create_rc_state_subdirectory_app_segment(tmp_path: Path) -> None:
    app_dir = get_or_create_rc_state_subdirectory(APP, home=tmp_path)

    assert app_dir == tmp_path / ".result-companion" / APP
    assert app_dir.is_dir()
