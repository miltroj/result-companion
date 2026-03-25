from pathlib import Path

from result_companion.core.rc_paths import (
    BUNDLED_DEFAULT_CONFIG,
    resolve_user_config,
)


def test_resolve_user_config_prefers_cli_config():
    cli_path = Path("/some/custom/config.yaml")

    result = resolve_user_config(cli_config=cli_path)

    assert result == cli_path


def test_resolve_user_config_falls_back_to_user_dir(tmp_path, monkeypatch):
    user_cfg = tmp_path / "config.yaml"
    user_cfg.write_text("model: test")
    monkeypatch.setattr("result_companion.core.rc_paths.USER_CONFIG", user_cfg)

    result = resolve_user_config(cli_config=None)

    assert result == user_cfg


def test_resolve_user_config_returns_none_when_no_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "result_companion.core.rc_paths.USER_CONFIG", tmp_path / "nope.yaml"
    )

    result = resolve_user_config(cli_config=None)

    assert result is None


def test_bundled_default_config_exists():
    assert (
        BUNDLED_DEFAULT_CONFIG.exists()
    ), f"Bundled config missing at {BUNDLED_DEFAULT_CONFIG}"
